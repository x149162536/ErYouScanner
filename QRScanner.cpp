// windows.h 的 min/max 宏会让 std::min/std::max 无法编译（之前 L89 就是这么挂的），
// 必须在任何 include 之前关掉。
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "QRScanner.h"

#include <algorithm>
#include <cstdlib>

namespace
{
// 1.5.3 的中心 ROI 是 1280x720 上固定的 232x232，约等于屏幕高度的 32%。
// 这里按高度比例算，1280x720 上正好落到 232；如果输入是更大的原生帧，
// 中心块会等比放大，相对覆盖范围不变。
constexpr int kCenterMin = 232;
constexpr int kCenterMax = 512;
constexpr double kCenterRatio = 0.32;

// 1.5.3 的瓦片：w/3+128, h/3+128，128 像素重叠。
constexpr int kTileMargin = 128;

bool isGray(const cv::Mat& m)
{
	return !m.empty() && m.type() == CV_8UC1;
}

int centerSizeFor(const cv::Mat& img)
{
	if (img.empty())
	{
		return 0;
	}
	int size = static_cast<int>(img.rows * kCenterRatio);
	size = (std::clamp)(size, kCenterMin, kCenterMax);
	return (std::min)(size, (std::min)(img.cols, img.rows));
}
}

QRScanner::QRScanner()
{
	// 与 1.5.3 的同源配置一致：只找 QR、关 tryHarder、关内部降采样、
	// 最多 1 个符号、全局直方图二值化。
	zxOptions.setFormats(ZXing::BarcodeFormat::QRCode);
	zxOptions.setTryHarder(false);
	zxOptions.setTryRotate(true);
	zxOptions.setTryInvert(true);
	zxOptions.setTryDownscale(false);
	zxOptions.setMaxNumberOfSymbols(1);
	zxOptions.setBinarizer(ZXing::Binarizer::GlobalHistogram);

	// 与 1.5.3 一样允许用环境变量覆盖两个兜底节奏；不设则用 150/200ms。
	if (const char* v = std::getenv("SCANNER_TILE_MS"))
	{
		const int n = std::atoi(v);
		if (n > 0)
		{
			tileIntervalMs_ = n;
		}
	}
	if (const char* v = std::getenv("SCANNER_DEEP_MS"))
	{
		const int n = std::atoi(v);
		if (n > 0)
		{
			deepIntervalMs_ = n;
		}
	}
}

QRScanner::~QRScanner() = default;

cv::Mat QRScanner::toGray(const cv::Mat& img)
{
	if (img.empty())
	{
		return {};
	}
	if (img.channels() == 1)
	{
		return img;
	}

	cv::Mat gray;
	if (img.channels() == 4)
	{
		cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
	}
	else if (img.channels() == 3)
	{
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	}
	return gray;
}

bool QRScanner::decodeView(const cv::Mat& gray, const cv::Rect& roi, std::string& qrCode)
{
	if (!isGray(gray))
	{
		return false;
	}

	const cv::Rect bounds(0, 0, gray.cols, gray.rows);
	const cv::Rect r = roi & bounds;
	if (r.width < 16 || r.height < 16)
	{
		return false;
	}

	// 零拷贝 ROI：只改 data 指针/宽高，不做 resize、不做 cvtColor。
	const cv::Mat view = gray(r);
	const ZXing::ImageView image(view.data, view.cols, view.rows, ZXing::ImageFormat::Lum,
	                             static_cast<int>(view.step), 1);
	const ZXing::Barcode result = ZXing::ReadBarcode(image, zxOptions);
	if (!result.isValid())
	{
		return false;
	}

	qrCode = result.text();
	return !qrCode.empty();
}

bool QRScanner::decodeCenter(const cv::Mat& img, std::string& qrCode)
{
	if (img.empty())
	{
		return false;
	}

	const int size = centerSizeFor(img);
	if (size < 16)
	{
		return false;
	}

	const int x = (img.cols - size) / 2;
	const int y = (img.rows - size) / 2;
	const cv::Rect roi(x, y, size, size);

	// 只把中心块转成灰度；命中时完全不碰全屏灰度化。
	const cv::Mat center = img(roi);
	cv::Mat centerGray;
	if (center.channels() == 1)
	{
		centerGray = center;
	}
	else if (center.channels() == 4)
	{
		cv::cvtColor(center, centerGray, cv::COLOR_BGRA2GRAY);
	}
	else if (center.channels() == 3)
	{
		cv::cvtColor(center, centerGray, cv::COLOR_BGR2GRAY);
	}
	else
	{
		return false;
	}

	return decodeView(centerGray, cv::Rect(0, 0, centerGray.cols, centerGray.rows), qrCode);
}

bool QRScanner::decodeHalf(const cv::Mat& gray, std::string& qrCode)
{
	if (!isGray(gray) || gray.cols < 64 || gray.rows < 64)
	{
		return false;
	}

	const int w = gray.cols / 2;
	const int h = gray.rows / 2;

	// 隔行隔列采样：pixStride=2、rowStride=step*2，等价于半分辨率的零拷贝视图。
	const ZXing::ImageView image(gray.data, w, h, ZXing::ImageFormat::Lum,
	                             static_cast<int>(gray.step) * 2, 2);
	const ZXing::Barcode result = ZXing::ReadBarcode(image, zxOptions);
	if (!result.isValid())
	{
		return false;
	}

	qrCode = result.text();
	return !qrCode.empty();
}

bool QRScanner::decodeTiles(const cv::Mat& gray, std::string& qrCode)
{
	if (!isGray(gray))
	{
		return false;
	}

	const int tileW = (std::min)(gray.cols, gray.cols / 3 + kTileMargin);
	const int tileH = (std::min)(gray.rows, gray.rows / 3 + kTileMargin);
	if (tileW < 32 || tileH < 32)
	{
		return false;
	}
	if (tileW >= gray.cols && tileH >= gray.rows)
	{
		return decodeFull(gray, qrCode);
	}

	// 左/中/右、上/中/下，瓦片之间重叠 128px，避免二维码被切断。
	const int xs[3] = { 0, (gray.cols - tileW) / 2, gray.cols - tileW };
	const int ys[3] = { 0, (gray.rows - tileH) / 2, gray.rows - tileH };

	for (int yi = 0; yi < 3; ++yi)
	{
		for (int xi = 0; xi < 3; ++xi)
		{
			if (decodeView(gray, cv::Rect(xs[xi], ys[yi], tileW, tileH), qrCode))
			{
				return true;
			}
		}
	}
	return false;
}

bool QRScanner::decodeFull(const cv::Mat& gray, std::string& qrCode)
{
	if (!isGray(gray))
	{
		return false;
	}
	return decodeView(gray, cv::Rect(0, 0, gray.cols, gray.rows), qrCode);
}

void QRScanner::decodeSingle(const cv::Mat& img, std::string& qrCode)
{
	qrCode.clear();
	if (img.empty())
	{
		return;
	}

	if (!timersInit_)
	{
		const auto now = std::chrono::steady_clock::now();
		lastTile_ = now - std::chrono::seconds(1);
		lastDeep_ = now - std::chrono::seconds(1);
		timersInit_ = true;
	}

	// 1) 每帧：中心灰度块（1.5.3 的快路径）
	if (decodeCenter(img, qrCode))
	{
		return;
	}

	// 2) 每帧：半分辨率全屏（1.5.3 的 fullScanLoop 第一层）
	const cv::Mat gray = toGray(img);
	if (gray.empty())
	{
		return;
	}
	if (decodeHalf(gray, qrCode))
	{
		return;
	}

	const auto now = std::chrono::steady_clock::now();

	// 3) 每 150ms：3x3 重叠瓦片（1.5.3 的 SCANNER_TILE_MS 默认值）
	if (now - lastTile_ >= std::chrono::milliseconds(tileIntervalMs_))
	{
		lastTile_ = now;
		if (decodeTiles(gray, qrCode))
		{
			return;
		}
	}

	// 4) 每 200ms：全分辨率 deep（1.5.3 的 SCANNER_DEEP_MS 默认值）
	if (now - lastDeep_ >= std::chrono::milliseconds(deepIntervalMs_))
	{
		lastDeep_ = now;
		if (decodeFull(gray, qrCode))
		{
			return;
		}
	}
}

void QRScanner::decodeMultiple(const cv::Mat& img, std::string& qrCode)
{
	decodeSingle(img, qrCode);
}
