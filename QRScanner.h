#pragma once

#include <chrono>
#include <mutex>
#include <string>

#include <opencv2/opencv.hpp>

#include <ReadBarcode.h>

// 按 MHY_Scanner 1.5.3 的扫码工作点实现（纯 zxing，无 CNN）：
//   1) 每帧先解屏幕正中心 232x232 灰度块（快路径，只转这一小块灰度）
//   2) 中心没解出 -> 640x360 半分辨率全屏（隔行隔列 stride 视图，零拷贝）
//   3) 每 150ms -> 3x3 重叠瓦片，原生分辨率
//   4) 每 200ms -> 全分辨率 deep
class QRScanner
{
public:
	QRScanner();
	~QRScanner();

	void decodeSingle(const cv::Mat& img, std::string& qrCode);
	void decodeMultiple(const cv::Mat& img, std::string& qrCode);

private:
	// 在 gray 的 roi 上做一次 zxing 解码；roi 会自动裁剪到图像范围内。
	bool decodeView(const cv::Mat& gray, const cv::Rect& roi, std::string& qrCode);
	// 中心灰度块（输入可以是 BGRA/BGR/Lum；只有这一小块会转灰度）。
	bool decodeCenter(const cv::Mat& img, std::string& qrCode);
	// 半分辨率全屏（pixStride=2, rowStride=step*2，零拷贝）。
	bool decodeHalf(const cv::Mat& gray, std::string& qrCode);
	// 3x3 重叠瓦片，瓦片尺寸 w/3+128, h/3+128。
	bool decodeTiles(const cv::Mat& gray, std::string& qrCode);
	// 全分辨率兜底。
	bool decodeFull(const cv::Mat& gray, std::string& qrCode);

	static cv::Mat toGray(const cv::Mat& img);

	ZXing::ReaderOptions zxOptions;
	int tileIntervalMs_{ 150 };
	int deepIntervalMs_{ 200 };
	std::chrono::steady_clock::time_point lastTile_{};
	std::chrono::steady_clock::time_point lastDeep_{};
	bool timersInit_{ false };
	// 采集池可能并发调用同一个 scanner，串行化，避免瓦片/deep 计时器互相踩。
	std::mutex mutex_;
};
