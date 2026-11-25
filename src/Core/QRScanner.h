#pragma once

#include <string>

#include <opencv2/wechat_qrcode.hpp>
#include <opencv2/opencv.hpp>

#include <ReadBarcode.h>

class QRScanner
{
public:
	QRScanner();
	~QRScanner();
	void decodeSingle(const cv::Mat& img, std::string& qrCode);
	void decodeMultiple(const cv::Mat& img, std::string& qrCode);

private:
	// zxing-cpp 经典算法快路径（非神经网络，毫秒级）
	bool decodeFast(const cv::Mat& img, std::string& qrCode);

	cv::Ptr<cv::wechat_qrcode::WeChatQRCode> detector; // CNN 兜底
	ZXing::ReaderOptions zxOptions;                    // zxing 配置（跨帧复用）
	int cnnMissStreak{ 0 };                            // zxing 连续失败计数，控制 CNN 兜底频率
};
