#include "QRCodeForScreen.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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

// 1.5.3 的采集端：一趟 pass 直接把 BGRA 帧缩成 1280x720 灰度。
// 原来这里是 cv::resize 到 BGRA（每帧写 3.69MB），扫描端再每帧/miss 转一次灰度；
// 现在只写 0.92MB，中心 crop、半分辨率、瓦片全部在这张灰度图上零拷贝。
void downscaleGray(const unsigned char* src, int srcW, int srcH, int srcStride, cv::Mat& dst)
{
    constexpr int kDstW = 1280;
    constexpr int kDstH = 720;
    if (!src || srcW <= 0 || srcH <= 0 || srcStride < srcW * 4)
    {
        dst.release();
        return;
    }

    dst.create(kDstH, kDstW, CV_8UC1);
    const double sx = static_cast<double>(srcW) / kDstW;
    const double sy = static_cast<double>(srcH) / kDstH;
    const int spanX = (std::max)(1, (std::min)(2, static_cast<int>(std::lround(sx))));
    const int spanY = (std::max)(1, (std::min)(2, static_cast<int>(std::lround(sy))));

    cv::parallel_for_(cv::Range(0, kDstH), [&](const cv::Range& range)
    {
        for (int y = range.start; y < range.end; ++y)
        {
            const int y0 = (std::min)(srcH - 1, static_cast<int>(y * sy));
            const int y1 = (std::min)(srcH - 1, y0 + spanY - 1);
            const unsigned char* row0 = src + static_cast<std::size_t>(y0) * srcStride;
            const unsigned char* row1 = src + static_cast<std::size_t>(y1) * srcStride;
            unsigned char* out = dst.ptr<unsigned char>(y);
            for (int x = 0; x < kDstW; ++x)
            {
                const int x0 = (std::min)(srcW - 1, static_cast<int>(x * sx));
                const int x1 = (std::min)(srcW - 1, x0 + spanX - 1);
                const unsigned char* p00 = row0 + static_cast<std::size_t>(x0) * 4;
                const unsigned char* p01 = row0 + static_cast<std::size_t>(x1) * 4;
                const unsigned char* p10 = row1 + static_cast<std::size_t>(x0) * 4;
                const unsigned char* p11 = row1 + static_cast<std::size_t>(x1) * 4;
                const unsigned b = p00[0] + p01[0] + p10[0] + p11[0];
                const unsigned g = p00[1] + p01[1] + p10[1] + p11[1];
                const unsigned r = p00[2] + p01[2] + p10[2] + p11[2];
                out[x] = static_cast<unsigned char>((306u * r + 601u * g + 117u * b + 2048u) >> 12);
            }
        }
    });
}

struct WaitForScreenWorkers
{
    QThreadPool& pool;
    std::atomic<bool>& running;
    ~WaitForScreenWorkers()
    {
        running.store(false);
        pool.waitForDone();
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
    QThreadPool pool;
    pool.setMaxThreadCount(threadNumber);
    WaitForScreenWorkers waitForWorkers{pool, m_stop};
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
            // 直接产出 1280x720 灰度：扫描端不再做任何颜色转换，也不再拷贝 3.69MB BGRA。
            downscaleGray(buffer.data(), width, height, width * 4, image);
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
            pool.tryStart([this, &scanner, image = std::move(image)]() {
                try
                {
                    if (!m_stop.load() || isInterruptionRequested())
                    {
                        return;
                    }
                    std::string decoded;
                    scanner.decodeSingle(image, decoded);
                    if (kuroAccount)
                    {
                        if (!m_stop.load() || !KuroApi::isGameQr(decoded)) return;
                        const auto status = kuroSession.prepare(decoded, resultGeneration);
                        if (!m_stop.load() || isInterruptionRequested()) return;
                        if (status == ScanRet::UNKNOW) return;
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
                        return;
                    }
                    const std::string passportQrUrl = PandaScanQRCode(scanUrl, ticket, gameType);
                    const bool accepted = !passportQrUrl.empty();
                    if (accepted)
                    {
                        lastQrCode = passportQrUrl;
                    }
                    if (!m_stop.load() || isInterruptionRequested())
                    {
                        return;
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
                }
                catch (...)
                {
                    LogScannerException("screen: QR worker");
                    stop();
                    Q_EMIT loginResults(ScanRet::STREAMERROR, resultGeneration);
                }
            });
        }
        // 有新帧：按识别间隔（可低到 1ms）；没新帧：睡 200ms，别空转重解同一帧。
        const int sleepMs = (result == 0) ? m_scanIntervalMs.load() : kIdleIntervalMs;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }
    const bool reportCaptureFailure = captureFailed && m_stop.exchange(false);
    pool.waitForDone();
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
