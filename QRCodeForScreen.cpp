#include "QRCodeForScreen.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include <QThreadPool>

#include "QRScanner.h"
#include "ScannerTask.hpp"
#include "ScreenShotDXGI.hpp"

namespace
{
// 1.5.3 的采集循环末尾也有一条长 Sleep（0x14015682e）：画面没变化时不空转。
constexpr int kIdleIntervalMs = 200;
// SCANNER_DEBUG=1：把每次命中的路径/耗时写进 scanner.log。
const bool kScanDebug = std::getenv("SCANNER_DEBUG") != nullptr;
// SCANNER_BGRA=1：跳过自写灰度化，回到 OpenCV resize + cvtColor 的旧路径（排障用）。
const bool kScanBgra = std::getenv("SCANNER_BGRA") != nullptr;

// 1.5.3 的采集端：一趟 pass 直接把 BGRA 帧缩成 1280x720 灰度。
// 原来这里是 cv::resize 到 BGRA（每帧写 3.69MB），扫描端再每帧/miss 转一次灰度。
// 现在按面积加权采样（2x 缩放时正好是 2x2 平均，1.5x 时按覆盖比例加权），
// 输出 0.92MB 单通道，中心 crop、半分辨率、瓦片全部零拷贝。
struct AxisTaps
{
    int i0 = 0;
    int i1 = 0;
    double w0 = 1.0;
    double w1 = 0.0;
};

void buildTaps(int dstN, int srcN, std::vector<AxisTaps>& taps)
{
    taps.resize(static_cast<std::size_t>(dstN));
    const double scale = static_cast<double>(srcN) / dstN;
    for (int d = 0; d < dstN; ++d)
    {
        const double a = d * scale;
        const double b = (d + 1) * scale;
        if (scale <= 1.0)
        {
            const int i = (std::min)(srcN - 1, static_cast<int>(a));
            taps[d] = { i, i, 1.0, 0.0 };
            continue;
        }
        const int first = (std::min)(srcN - 1, static_cast<int>(std::floor(a)));
        const int last = (std::min)(srcN - 1, static_cast<int>(std::ceil(b)) - 1);
        if (last <= first)
        {
            taps[d] = { first, first, 1.0, 0.0 };
            continue;
        }
        const double c0 = (first + 1) - a;
        const double c1 = b - last;
        const double sum = (std::max)(1e-9, c0 + c1);
        taps[d] = { first, last, c0 / sum, c1 / sum };
    }
}

inline unsigned lumaOf(const unsigned char* p)
{
    return (306u * p[2] + 601u * p[1] + 117u * p[0] + 512u) >> 10;
}

void downscaleGray(const unsigned char* src, int srcW, int srcH, int srcStride, cv::Mat& dst)
{
    constexpr int kDstW = 1280;
    constexpr int kDstH = 720;
    if (!src || srcW <= 0 || srcH <= 0 || srcStride < srcW * 4)
    {
        dst.release();
        return;
    }

    // 分辨率变化时重建采样表；同一分辨率下复用。
    static int cachedW = -1;
    static int cachedH = -1;
    static std::vector<AxisTaps> tapX;
    static std::vector<AxisTaps> tapY;
    static std::mutex tapMutex;
    {
        const std::lock_guard<std::mutex> lock(tapMutex);
        if (cachedW != srcW || cachedH != srcH)
        {
            buildTaps(kDstW, srcW, tapX);
            buildTaps(kDstH, srcH, tapY);
            cachedW = srcW;
            cachedH = srcH;
        }
    }

    dst.create(kDstH, kDstW, CV_8UC1);
    cv::parallel_for_(cv::Range(0, kDstH), [&](const cv::Range& range)
    {
        for (int y = range.start; y < range.end; ++y)
        {
            const AxisTaps& ty = tapY[static_cast<std::size_t>(y)];
            const unsigned char* row0 = src + static_cast<std::size_t>(ty.i0) * srcStride;
            const unsigned char* row1 = src + static_cast<std::size_t>(ty.i1) * srcStride;
            unsigned char* out = dst.ptr<unsigned char>(y);
            for (int x = 0; x < kDstW; ++x)
            {
                const AxisTaps& tx = tapX[static_cast<std::size_t>(x)];
                const unsigned char* p00 = row0 + static_cast<std::size_t>(tx.i0) * 4;
                const unsigned char* p10 = row0 + static_cast<std::size_t>(tx.i1) * 4;
                const unsigned char* p01 = row1 + static_cast<std::size_t>(tx.i0) * 4;
                const unsigned char* p11 = row1 + static_cast<std::size_t>(tx.i1) * 4;

                const double top = tx.w0 * lumaOf(p00) + tx.w1 * lumaOf(p10);
                const double bottom = tx.w0 * lumaOf(p01) + tx.w1 * lumaOf(p11);
                const double value = ty.w0 * top + ty.w1 * bottom;
                out[x] = static_cast<unsigned char>((std::min)(255.0, (std::max)(0.0, value + 0.5)));
            }
        }
    });
}

// 采集线程只发布“最新一帧”，扫描线程永远取最新的：不排队、不丢中间帧。
struct FrameSlot
{
    std::mutex mutex;
    std::condition_variable cv;
    cv::Mat frame;
    bool closed = false;

    void push(cv::Mat next)
    {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            frame = std::move(next);
        }
        cv.notify_one();
    }

    bool pop(cv::Mat& out)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return closed || !frame.empty(); });
        if (closed)
        {
            return false;
        }
        out = std::move(frame);
        frame.release();
        return true;
    }

    void close()
    {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            closed = true;
        }
        cv.notify_all();
    }
};

struct WorkerJoiner
{
    FrameSlot& slot;
    std::thread& worker;
    ~WorkerJoiner()
    {
        slot.close();
        if (worker.joinable())
        {
            worker.join();
        }
    }
};
}

QRCodeForScreen::QRCodeForScreen(QObject* parent) : QThread(parent), m_stop(false)
{
    m_config = &ConfigDate::getInstance();
}

QRCodeForScreen::~QRCodeForScreen()
{
    stop();
    requestInterruption();
    wait();
}

void QRCodeForScreen::setScanInterval(int intervalMs)
{
    m_scanIntervalMs.store(std::max(1, intervalMs));
}

void QRCodeForScreen::setLoginInfo1(const std::string& uid, const std::string& stoken, const std::string& mid)
{
    kuroAccount = false;
    authorizationCancelled.store(false);
    lastTicket.clear();
    lastQrCode.clear();
    this->uid = uid;
    this->gameToken = stoken;
    this->mid = mid;
}

void QRCodeForScreen::setKuroLoginInfo(const std::string& uid, const std::string& token)
{
    this->uid = uid;
    kuroAccount = true;
    authorizationCancelled.store(false);
    kuroSession.setAccount(token, resultGeneration);
}

void QRCodeForScreen::LoginOfficial()
{
    monitorScreen();
}

void QRCodeForScreen::monitorScreen()
{
    // One worker owns decoder and login state; it is drained before either is destroyed.
    QRScanner scanner;
    ScreenShotDXGI capture;
    int width = 0;
    int height = 0;
    if (!capture.InitDevice() || !capture.InitDupl(0, width, height))
    {
        WriteScannerLog("screen: initialization failed hr=" + std::to_string(capture.lastError()));
        stop();
        Q_EMIT loginResults(ScanRet::STREAMERROR, resultGeneration);
        return;
    }
    const auto maximum = (std::numeric_limits<std::size_t>::max)();
    if (width <= 0 || height <= 0 || static_cast<std::size_t>(width) > maximum / 4 ||
        static_cast<std::size_t>(height) > maximum / (static_cast<std::size_t>(width) * 4))
    {
        throw std::runtime_error("Invalid desktop frame size");
    }
    std::vector<unsigned char> buffer(static_cast<std::size_t>(width) * height * 4);
    DXGI_FORMAT lastFormat = DXGI_FORMAT_UNKNOWN;
    cv::Mat lastGoodFrame;
    FrameSlot slot;
    std::thread worker([this, &scanner, &slot]()
    {
        while (true)
        {
            cv::Mat image;
            if (!slot.pop(image))
            {
                return;
            }
            try
            {
                if (!m_stop.load() || isInterruptionRequested())
                {
                    return;
                }
                std::string decoded;
                scanner.decodeSingle(image, decoded);
                if (kScanDebug && !decoded.empty())
                {
                    WriteScannerLog("qr: path=" + scanner.lastPath() +
                        " us=" + std::to_string(scanner.lastMicros()) +
                        " frame=" + std::to_string(image.cols) + "x" + std::to_string(image.rows));
                }
                if (kuroAccount)
                {
                    if (!m_stop.load() || !KuroApi::isGameQr(decoded)) continue;
                    const auto status = kuroSession.prepare(decoded, resultGeneration);
                    if (!m_stop.load() || isInterruptionRequested()) continue;
                    if (status == ScanRet::UNKNOW) continue;
                    stop();
                    if (status == ScanRet::SUCCESS) Q_EMIT loginConfirm(GameType::WutheringWaves, true, resultGeneration);
                    else Q_EMIT loginResults(status, resultGeneration);
                    return;
                }
                std::string ticket;
                if (!m_stop.load() || isInterruptionRequested() ||
                    !parseOfficialQRCode(decoded, ticket) ||
                    lastTicket == ticket)
                {
                    continue;
                }
                const std::string passportQrUrl = PandaScanQRCode(scanUrl, ticket, gameType);
                const bool accepted = !passportQrUrl.empty();
                if (accepted)
                {
                    lastQrCode = passportQrUrl;
                }
                if (!m_stop.load() || isInterruptionRequested())
                {
                    continue;
                }
                if (accepted)
                {
                    lastTicket = ticket;
                    // Stop capture before publishing a confirmation to the UI.
                    stop();
                    Q_EMIT loginConfirm(gameType, true, resultGeneration);
                }
                else
                {
                    stop();
                    Q_EMIT loginResults(ScanRet::FAILURE_1, resultGeneration);
                }
                return;
            }
            catch (...)
            {
                LogScannerException("screen: QR worker");
                stop();
                Q_EMIT loginResults(ScanRet::STREAMERROR, resultGeneration);
                return;
            }
        }
    });
    WorkerJoiner joiner{ slot, worker };
    bool captureFailed = false;
    while (m_stop.load() && !isInterruptionRequested())
    {
        const int result = capture.getFrame(100);
        if (result == 1)
        {
            WriteScannerLog("screen: acquisition failed hr=" + std::to_string(capture.lastError()));
            captureFailed = true;
            break;
        }
        cv::Mat image;
        if (result == 0)
        {
            const auto desc = capture.frameDescription();
            if (!desc.Width || !desc.Height || desc.Width > INT_MAX || desc.Height > INT_MAX ||
                desc.Width > maximum / 4 || desc.Height > maximum / (static_cast<std::size_t>(desc.Width) * 4))
            {
                WriteScannerLog("screen: invalid acquired frame dimensions");
                capture.doneWithFrame();
                captureFailed = true;
                break;
            }
            if (lastFormat != desc.Format || width != static_cast<int>(desc.Width) || height != static_cast<int>(desc.Height))
            {
                WriteScannerLog("screen: frame width=" + std::to_string(desc.Width) +
                    " height=" + std::to_string(desc.Height) + " format=" + std::to_string(desc.Format));
                lastFormat = desc.Format;
            }
            width = static_cast<int>(desc.Width);
            height = static_cast<int>(desc.Height);
            buffer.resize(static_cast<std::size_t>(width) * height * 4);
            if (!capture.copyFrameToBuffer(buffer.data(), buffer.size()))
            {
                WriteScannerLog("screen: frame copy failed hr=" + std::to_string(capture.lastError()) +
                    " stage=" + capture.lastCopyStage() + " format=" + std::to_string(desc.Format) +
                    " width=" + std::to_string(width) + " height=" + std::to_string(height) +
                    " rowPitch=" + std::to_string(capture.lastRowPitch()) + " capacity=" + std::to_string(buffer.size()));
                capture.doneWithFrame();
                captureFailed = true;
                break;
            }
            // Release the DXGI frame before resize/decode, including exception paths.
            if (!capture.doneWithFrame())
            {
                WriteScannerLog("screen: frame release failed hr=" + std::to_string(capture.lastError()));
                captureFailed = true;
                break;
            }
            if (kScanBgra || width > 3000)
            {
                // OpenCV 路径：4K（3x 以上降采样，自写 2-tap 会混叠）或排障回退。
                const cv::Mat bgra(height, width, CV_8UC4, buffer.data());
                if (kScanBgra)
                {
                    cv::Mat small;
                    cv::resize(bgra, small, { 1280, 720 });
                    cv::cvtColor(small, image, cv::COLOR_BGRA2GRAY);
                }
                else
                {
                    cv::Mat full;
                    cv::cvtColor(bgra, full, cv::COLOR_BGRA2GRAY);
                    cv::resize(full, image, { 1280, 720 }, 0, 0, cv::INTER_AREA);
                }
            }
            else
            {
                // 直接产出 1280x720 灰度：扫描端不再做任何颜色转换，也不再拷贝 3.69MB BGRA。
                downscaleGray(buffer.data(), width, height, width * 4, image);
            }
            lastGoodFrame = image;
        }
        else if (!lastGoodFrame.empty())
        {
            image = lastGoodFrame;
        }
        if (!image.empty())
        {
#ifdef ERYOU_SCAN_PREVIEW
            cv::imshow("Video_Stream", image);
            cv::waitKey(1);
#endif
            slot.push(std::move(image));
        }
        // 有新帧：按识别间隔（可低到 1ms）；没新帧：睡 200ms，别空转重解同一帧。
        const int sleepMs = (result == 0) ? m_scanIntervalMs.load() : kIdleIntervalMs;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }
    slot.close();
    if (worker.joinable())
    {
        worker.join();
    }
    const bool reportCaptureFailure = captureFailed && m_stop.exchange(false);
    if (reportCaptureFailure)
    {
        Q_EMIT loginResults(ScanRet::STREAMERROR, resultGeneration);
    }
}

void QRCodeForScreen::continueLastLogin()
{
    if (authorizationCancelled.load()) return;
    if (kuroAccount)
    {
        Q_EMIT loginResults(kuroSession.confirm({}, resultGeneration), resultGeneration);
        return;
    }
    const bool loggedIn = ScanPassportQRLogin(lastQrCode, gameToken, mid) && !authorizationCancelled.load() &&
                          ConfirmPassportQRLogin(lastQrCode, gameToken, mid);
    Q_EMIT loginResults(loggedIn ? ScanRet::SUCCESS : ScanRet::FAILURE_2, resultGeneration);
}

void QRCodeForScreen::run()
{
    ret = ScanRet::UNKNOW;
    m_stop.store(true);
    try
    {
#ifdef ERYOU_SCAN_PREVIEW
        cv::namedWindow("Video_Stream", cv::WINDOW_AUTOSIZE);
#endif
        LoginOfficial();
    }
    catch (...)
    {
        LogScannerException("screen: model or capture initialization");
        Q_EMIT loginResults(ScanRet::STREAMERROR, resultGeneration);
    }
    m_stop.store(false);
#ifdef ERYOU_SCAN_PREVIEW
    try
    {
        cv::destroyWindow("Video_Stream");
    }
    catch (...)
    {
        LogScannerException("screen: preview cleanup");
    }
#endif
}

void QRCodeForScreen::stop()
{
    requestInterruption();
    m_stop.store(false);
}
