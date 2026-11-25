#include "QRScanner.h"

#include <algorithm>

#include "QRModelPaths.hpp"

// 快路径输入的最长边；超过则先缩小再交给 zxing
static constexpr int FAST_MAX_EDGE = 1280;
// zxing 连续失败达到该次数后才启用 CNN 兜底，之后每 N 帧兜底一次，避免 CNN 阻塞扫描线程
static constexpr int CNN_FALLBACK_EVERY = 3;

namespace
{
// 对给定图像做一次 zxing 解码（支持灰度 / BGR / BGRA）
bool decodeMat(const cv::Mat& m, const ZXing::ReaderOptions& options, std::string& qrCode)
{
    if (m.empty() || m.depth() != CV_8U)
    {
        return false;
    }

    ZXing::ImageFormat fmt;
    int pixStride = 0;
    switch (m.channels())
    {
    case 1: fmt = ZXing::ImageFormat::Lum;  pixStride = 1; break;
    case 3: fmt = ZXing::ImageFormat::BGR;  pixStride = 3; break;
    case 4: fmt = ZXing::ImageFormat::BGRA; pixStride = 4; break;
    default: return false;
    }

    const ZXing::ImageView view(m.data, m.cols, m.rows, fmt,
                                static_cast<int>(m.step), pixStride);
    const ZXing::Barcode result = ZXing::ReadBarcode(view, options);
    if (!result.isValid())
    {
        return false;
    }
    qrCode = result.text();
    return !qrCode.empty();
}
}

QRScanner::QRScanner()
{
    const auto paths = QRModelPaths(QRModelDirectory());
    detector = cv::makePtr<cv::wechat_qrcode::WeChatQRCode>(paths[0].string(), paths[1].string(),
                                                          paths[2].string(), paths[3].string());
    // CNN 只做兜底，缩小检测输入以降低单帧耗时（原版为 1.0）
    detector->setScaleFactor(0.5);

    // zxing-cpp：只认二维码，开旋转/反色，关降采样（我们自己按 FAST_MAX_EDGE 缩），用全局直方图二值化
    zxOptions.setFormats(ZXing::BarcodeFormat::QRCode);
    zxOptions.setTryHarder(false);
    zxOptions.setTryRotate(true);
    zxOptions.setTryInvert(true);
    zxOptions.setTryDownscale(false);
    zxOptions.setMaxNumberOfSymbols(1);
    zxOptions.setBinarizer(ZXing::Binarizer::GlobalHistogram);
}

QRScanner::~QRScanner()
{
}

// 比“直接全屏喂 zxing”更快：先只解中心区域（游戏登录码基本在屏幕中央），
// 全屏只在中心失败时兜一次，并且先缩小 + 转灰度，减少通道运算。
bool QRScanner::decodeFast(const cv::Mat& img, std::string& qrCode)
{
    if (img.empty() || img.depth() != CV_8U)
    {
        return false;
    }

    // 1) 中心 50%×50% 区域优先（面积只有全屏的 1/4）
    const int centerW = img.cols / 2;
    const int centerH = img.rows / 2;
    if (centerW >= 80 && centerH >= 80)
    {
        const cv::Mat center = img(cv::Rect((img.cols - centerW) / 2, (img.rows - centerH) / 2, centerW, centerH));
        if (decodeMat(center, zxOptions, qrCode))
        {
            return true;
        }
    }

    // 2) 全屏兜底：超长边先缩小，彩色先转灰度
    cv::Mat view;
    const double scale = static_cast<double>(FAST_MAX_EDGE) / std::max(img.cols, img.rows);
    if (scale < 1.0)
    {
        cv::resize(img, view, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    else
    {
        view = img;
    }

    if (view.channels() == 4)
    {
        cv::Mat gray;
        cv::cvtColor(view, gray, cv::COLOR_BGRA2GRAY);
        return decodeMat(gray, zxOptions, qrCode);
    }
    if (view.channels() == 3)
    {
        cv::Mat gray;
        cv::cvtColor(view, gray, cv::COLOR_BGR2GRAY);
        return decodeMat(gray, zxOptions, qrCode);
    }
    return decodeMat(view, zxOptions, qrCode);
}

void QRScanner::decodeSingle(const cv::Mat& img, std::string& qrCode)
{
    qrCode.clear();
    if (img.empty())
    {
        return;
    }

    // 1) 快路径：中心 ROI + zxing（干净的游戏/直播二维码通常 <1ms 就能解出）
    if (decodeFast(img, qrCode))
    {
        cnnMissStreak = 0;
        return;
    }

    // 2) 慢速兜底：WeChatQRCode（CNN）。连续失败 3 次才进，之后每 3 帧一次，
    //    避免每帧 80-100ms 的 CNN 把扫描线程堵死。
    ++cnnMissStreak;
    if (cnnMissStreak < CNN_FALLBACK_EVERY || cnnMissStreak % CNN_FALLBACK_EVERY != 0)
    {
        return;
    }

    const std::vector<std::string>& strDecoded = detector->detectAndDecode(img);
    if (!strDecoded.empty() && !strDecoded[0].empty())
    {
        qrCode = strDecoded[0];
        cnnMissStreak = 0;
    }
}

void QRScanner::decodeMultiple(const cv::Mat& img, std::string& qrCode)
{
    qrCode.clear();
    if (img.empty())
    {
        return;
    }

    if (decodeFast(img, qrCode))
    {
        cnnMissStreak = 0;
        return;
    }

    ++cnnMissStreak;
    if (cnnMissStreak < CNN_FALLBACK_EVERY || cnnMissStreak % CNN_FALLBACK_EVERY != 0)
    {
        return;
    }

    const std::vector<std::string>& strDecoded = detector->detectAndDecode(img);
    if (!strDecoded.empty())
    {
        qrCode = strDecoded.back();
        cnnMissStreak = 0;
    }
}
