/****************************************************************
	LUT読み込み＆画像変換プログラム
	Author: Nishizawa,	Date: 2015/08/25
	必要なライブラリ：OpenCV 3.0 (画像処理，行列演算等)

	【機能】
	PNGまたはXML形式のLUTを読み込み，LUTに従って画像変換します．
	但しXML形式は非常に重く，途中でオーバーフローしたり
	LUTを全て読み込めない場合があったりするので非推奨です．
*****************************************************************/

#include "OpenCV3Linker.h"

using namespace cv;
using namespace std;

//	LUTのアクセス用マクロ
#define BIT(B, G, R) ((B) << 16 | (G) << 8 | (R))

//--------------------------------------------
//	関数プロトタイプ
//--------------------------------------------
void lookup(Mat &src, Mat &dst, Mat &LUT);

//-------------------------
//	読み込んだLUTの保存場所
//-------------------------
Mat LUT_elder70;			//	70歳高齢者シミュレーション用
Mat LUT_elder80;			//	80歳高齢者シミュレーション用
Mat LUT_typeP;			//	一型(P)二色覚シミュレーション用
Mat LUT_typeD;			//	二型(D)二色覚シミュレーション用
Mat LUT_typeT;			//	三型(T)二色覚シミュレーション用

int main(void)
{
	//------------------------------------------
	//	PNG形式
	//------------------------------------------
	//	現在のTickCount
	double f = 1000.0 / cv::getTickFrequency();
	int64 time = cv::getTickCount();

	//	PNG形式LUTの読み込み
	cout << "PNG形式のLUTを読み込んでいます..." << endl;
	cout << "高齢者特性を読み込み中..." << endl;
	LUT_elder70 = imread("data/LUT_elder_70.png");
	LUT_elder80 = imread("data/LUT_elder_80.png");
	cout << "色覚特性を読み込み中..." << endl;
	LUT_typeP = imread("data/LUT_dichromat_typeP.png", CV_LOAD_IMAGE_COLOR);
	LUT_typeD = imread("data/LUT_dichromat_typeD.png", CV_LOAD_IMAGE_COLOR);
	LUT_typeT = imread("data/LUT_dichromat_typeT.png", CV_LOAD_IMAGE_COLOR);
	cout << "読み込みが終了しました." << endl;

	//	TickCountの変化を[ms]単位で表示
	std::cout << "処理時間: " << (cv::getTickCount() - time)*f << " [ms]" << std::endl;

	////------------------------------------------
	////	XML形式
	////	重くて読み込めないけど一応残しておく
	////------------------------------------------
	////	現在のTickCount
	//f = 1000.0 / cv::getTickFrequency();
	//time = cv::getTickCount();

	////	XML形式でLUTを読み込み
	//cout << "XML形式のLUTを読み込んでいます..." << endl;
	////	高齢者LUT
	//cout << "高齢者特性を読み込み中..." << endl;
	//FileStorage elderData("data/LUT_elder.xml", FileStorage::READ);
	//FileNode calibNode = elderData["calibration"];
	//FileNode gammaNode = calibNode["display_gamma"];
	//FileNode luminanceNode = calibNode["display_luminance"];
	//FileNode xyNode = calibNode["display_xy"];
	//FileNode lutNode = elderData["LUT"];
	//double gamma[3] = {
	//	static_cast<double>(gammaNode["gamma_b"]),
	//	static_cast<double>(gammaNode["gamma_g"]),
	//	static_cast<double>(gammaNode["gamma_r"]) 
	//};
	//double luminance[3] = {
	//	static_cast<double>(gammaNode["luminance_b"]),
	//	static_cast<double>(gammaNode["luminance_g"]),
	//	static_cast<double>(gammaNode["luminance_r"]) 
	//};
	//Point2d xyDisp[3];
	//xyNode["xy_b"] >> xyDisp[0];
	//xyNode["xy_g"] >> xyDisp[1];
	//xyNode["xy_r"] >> xyDisp[2];
	//lutNode["seventy"] >> LUT_elder70;
	//lutNode["eighty"] >> LUT_elder80;
	//elderData.release();
	//~calibNode;
	//~gammaNode;
	//~luminanceNode;
	//~xyNode;
	//~lutNode;
	////	色覚LUT
	//cout << "色覚特性を読み込み中..." << endl;
	//FileStorage dichromantData("data/LUT_elder.xml", FileStorage::READ);
	//calibNode = dichromantData["calibration"];
	//gammaNode = calibNode["display_gamma"];
	//luminanceNode = calibNode["display_luminance"];
	//xyNode = calibNode["display_xy"];
	//lutNode = dichromantData["LUT"];
	//gamma[0] = static_cast<double>(gammaNode["gamma_b"]);
	//gamma[1] = static_cast<double>(gammaNode["gamma_g"]);
	//gamma[2] = static_cast<double>(gammaNode["gamma_r"]);
	//luminance[0] = static_cast<double>(gammaNode["luminance_b"]);
	//luminance[1] = static_cast<double>(gammaNode["luminance_g"]);
	//luminance[2] = static_cast<double>(gammaNode["luminance_r"]);
	//xyNode["xy_b"] >> xyDisp[0];
	//xyNode["xy_g"] >> xyDisp[1];
	//xyNode["xy_r"] >> xyDisp[2];
	//lutNode["type_p"] >> LUT_typeP;
	//lutNode["type_d"] >> LUT_typeD;
	//lutNode["type_t"] >> LUT_typeT;
	//dichromantData.release();
	//~calibNode;
	//~gammaNode;
	//~luminanceNode;
	//~xyNode;
	//~lutNode;

	//cout << "読み込みが終了しました．" << endl;

	//// TickCountの変化を[ms]単位で表示
	//std::cout << "処理時間: " << (cv::getTickCount() - time)*f << " [ms]" << std::endl;

	//	元画像の読み込み
	Mat original = imread("img/lovelive.png");
	resize(original, original, Size(320, 240));

	//	LUTによる変換
	Mat elder70, elder80, typeP, typeD, typeT;
	lookup(original, elder70, LUT_elder70);
	lookup(original, elder80, LUT_elder80);
	lookup(original, typeP, LUT_typeP);
	lookup(original, typeD, LUT_typeD);
	lookup(original, typeT, LUT_typeT);

	//	表示
	imshow("元画像", original);
	imshow("70歳", elder70);
	imshow("80歳", elder80);
	imshow("1型2色覚", typeP);
	imshow("2型2色覚", typeD);
	imshow("3型2色覚", typeT);

	waitKey(0);

	destroyAllWindows();

	return 0;
}

//	LUTに従って画素値の入れ替え
void lookup(Mat &src, Mat &dst, Mat &LUT)
{
	dst = Mat(src.rows, src.cols, CV_8UC3);
	for (int y = 0; y < dst.rows; y++)
	{
		for (int x = 0; x < dst.cols; x++)
		{
			matB(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[0];
			matG(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[1];
			matR(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[2];
		}
	}
}