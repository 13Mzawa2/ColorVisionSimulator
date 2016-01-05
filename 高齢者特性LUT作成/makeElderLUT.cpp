/***********************************************************************
	高齢者特性シミュレーション用LUT作成プログラム ver. 0.2
	Author: Nishizawa	Date: 2016/01/05
	必要なライブラリ: OpenCV 3.0
	
	【機能】
	高齢者の見えを小田・村上の手法で再現するLUTを，
	PNG形式，XML形式の両方で作成します．
	但しXML形式は非常に重いのでPNG形式を推奨します．

	【参考文献】
	神戸修論．高齢者視覚と色覚異常の動画シミュレーション．
	小田修論．高齢者の色覚特性を考慮した演色性の評価方法と可視化．
	村上卒論．高齢者の視覚特性シミュレータに関する研究．
	岩元修論．視覚障害シミュレータにおける空間周波数特性模擬の精緻化．

	【注意事項】
	本プログラムはOpenCVの機能に準ずるため，
	全てのRGB関連はBGR順で記述してあります．
	（その他の色空間はそのままです）
	その為，XYZ - RGB変換等で行列がいつもと違いますので
	注意して読んでください．
************************************************************************/

#include "OpenCV3Linker.h"
#include <omp.h>

using namespace cv;
using namespace std;

//	LUTのアクセス用マクロ
#define BIT(B, G, R) (((int)(B) << 16) | ((int)(G) << 8) | (R))

//--------------------------------------------
//	関数プロトタイプ
//--------------------------------------------
Point3d operator*(Mat M, const Point3d& p);
Point3d cvtxyY2XYZ(Point3d xyY);
Point3d cvtxyY2XYZ(Point2d xy, double Y);
Point3d cvtXYZ2xyY(Point3d XYZ);
Point3d cvtXYZ2Lab(Point3d XYZ, Point3d XYZ0);
Point3d cvtLab2XYZ(Point3d Lab, Point3d XYZ0);
void lookup(Mat &src, Mat &dst, Mat &LUT);

//----------------------------------------------
//	カラーキャリブレーション計測値
//	ディスプレイのガンマ特性等をここに入力する
//----------------------------------------------
static double gamma[3] = { 2.43434386, 1.96794267, 1.762290873};		//	B, G, Rのガンマ係数
static double Lmax[3] = { 31.23, 452.13, 150.13 };						//	B, G, Rの最大出力輝度(階調値255の時の輝度)
//	ディスプレイの各蛍光体の平均色度
//	xyDisp[0].x = xB　のようになる
static Point2d xyDisp[3] = {
	Point2d(0.1375, 0.04131),	//	xB, yB
	Point2d(0.3301, 0.6022),	//	xG, yG
	Point2d(0.6232, 0.3787)		//	xR, yR
};
static Point3d xyYWhite(0.318, 0.335, 631);		//	白色板（完全拡散反射面）の測色値

//---------------------------------
//	高齢者シミュレーション用定数
//	神戸，小田の修論から引用
//---------------------------------
static Point3d K70(0.521, 0.673, 0.864);		//	70歳 vs 20歳の実効輝度比，BGR
static Point3d K80(0.388, 0.555, 0.847);		//	80歳 vs 20歳, BGR
static double alpha70 = 0.6, alpha80 = 0.5;			//	老人性縮瞳による網膜照度低下を表す視感透過率

//	YBGR to XYZのための行列　xyDisp[3]から計算する
static Mat cvtMatYBGR2XYZ = (Mat_<double>(3, 3) <<
	xyDisp[0].x / xyDisp[0].y, xyDisp[1].x / xyDisp[1].y, xyDisp[2].x / xyDisp[2].y,
	1.0, 1.0, 1.0,
	(1 - xyDisp[0].x - xyDisp[0].y) / xyDisp[0].y, (1 - xyDisp[1].x - xyDisp[1].y) / xyDisp[1].y, (1 - xyDisp[2].x - xyDisp[2].y) / xyDisp[2].y);
//	上記の逆行列
static Mat cvtMatXYZ2YBGR = cvtMatYBGR2XYZ.inv();
//	完全拡散反射面のXYZ
static Point3d XYZWhite = cvtxyY2XYZ(xyYWhite);

int main(void)
{
	//-------------------------------------------------
	//	ルックアップテーブルの作成
	//	b<<16 + g<<8 + r　のマクロでアクセスする
	//-------------------------------------------------
	Mat LUT_elder70(0xffffff + 1, 1, CV_8UC3);			//	70歳高齢者シミュレーション用
	Mat LUT_elder80(0xffffff + 1, 1, CV_8UC3);			//	80歳高齢者シミュレーション用
	cout << "高齢者特性LUTを作成中..." << endl;
#pragma omp parallel
	{
#pragma omp for
		for (int b = 0; b <= 0xff; b++)
		{
			static int counter = 1;
			cout << counter++ << " / 256 完了" << "\r";
			for (int g = 0; g <= 0xff; g++)
			{
				for (int r = 0; r <= 0xff; r++)
				{
					//	1. ガンマをもとに階調値BGRから表示デバイス出力輝度YBGRを求める
					//	YBGR.x = YG, YBGR.y = YG, YBGR.z = YR
					Point3d YBGR(	//	RGB蛍光体の出力輝度
						Lmax[0] * pow((double)b / 0xff, gamma[0]),		//	B
						Lmax[1] * pow((double)g / 0xff, gamma[1]),		//	G
						Lmax[2] * pow((double)r / 0xff, gamma[2])		//	R
						);
					//	2. 色恒常性を考慮した色相・彩度変換式の導入（小田・村上）
					//	注意：この変換式は平均24.4歳vs平均68.2歳の実験結果に基づく
					//	2.1. YBGR -> XYZ -> L*a*b*
					Point3d XYZ = cvtMatYBGR2XYZ * YBGR;
					Point3d Lab = cvtXYZ2Lab(XYZ, XYZWhite);
					//	2.2. 色相・彩度の変換係数を導出
					double chroma = atan2(Lab.y, Lab.x);		//	彩度
					double hue = sqrt(Lab.y * Lab.y + Lab.z * Lab.z);	//	色相[rad]
					double Cey = 1.32 * exp(-chroma / 6.53) - exp(-pow((chroma - 47.23) / 2680.53, 2)) / 27.89 + 1.0;
					double dth = 0.131 * cos(2 * CV_PI * (hue - 5.875) / 5.686) + 0.027;
					//	2.3. 高齢者色度変換
					Point3d Lab1(
						Lab.x,
						Cey * (cos(dth) * Lab.y - sin(dth) * Lab.z),
						Cey * (sin(dth) * Lab.y + cos(dth) * Lab.z)
						);
					//	2.4. L*a*b* -> XYZ -> YBGR
					Point3d XYZ1 = cvtLab2XYZ(Lab1, XYZWhite);
					Point3d YBGR1 = cvtMatXYZ2YBGR * XYZ;
					//	3. 老人性縮瞳を考慮した実効輝度比の計算（神戸）
					double Ke70 = alpha70 * (K70.x * YBGR1.x + K70.y * YBGR1.y + K70.z * YBGR1.z) / (YBGR1.x + YBGR1.y + YBGR1.z);
					double Ke80 = alpha80 * (K80.x * YBGR1.x + K80.y * YBGR1.y + K80.z * YBGR1.z) / (YBGR1.x + YBGR1.y + YBGR1.z);
					//	4. 実効輝度の計算
					//	Ye70.x = YeB(70歳変換後のB蛍光体の出力輝度), Ye70.y = YeG, Ye70.z = YeR
					Vec3d Ye70 = Ke70 * YBGR1;
					Vec3d Ye80 = Ke80 * YBGR1;
					//	4. ガンマ補正でBGRに逆算
					for (int i = 0; i < 3; i++)
					{
						double BGR70 = 0xff * pow(Ye70[i] / Lmax[i], 1.0 / gamma[i]);
						double BGR80 = 0xff * pow(Ye80[i] / Lmax[i], 1.0 / gamma[i]);

						LUT_elder70.at<Vec3b>(BIT(b, g, r), 0)[i] = (BGR70 > 0.0) ? ((BGR70 < 255.0) ? (uchar)BGR70 : 0xff) : 0x00;
						LUT_elder80.at<Vec3b>(BIT(b, g, r), 0)[i] = (BGR80 > 0.0) ? ((BGR80 < 255.0) ? (uchar)BGR80 : 0xff) : 0x00;
					}
				}
			}
		}		//	LUT作成終了
	}
	cout << "\nLUTの作成が終了しました．テスト画像を表示します．" << endl;
	//-----------------------------
	//	画像変換テスト
	//-----------------------------
	Mat testImg = imread("img/sapporo_new.png");
	Mat dstImg70, dstImg80;
	lookup(testImg, dstImg70, LUT_elder70);
	lookup(testImg, dstImg80, LUT_elder80);
	imshow("オリジナル", testImg);
	imshow("70歳変換", dstImg70);
	imshow("80歳変換", dstImg80);
	waitKey(0);
	destroyAllWindows();

	cout << "結果を保存中..." << endl;
	imwrite("img/sapporo_new70.png", dstImg70);
	imwrite("img/sapporo_new80.png", dstImg80);
	//-----------------------------
	//	LUTをPNG形式で保存
	//-----------------------------
	imwrite("data/LUT_elder_70.png", LUT_elder70);
	imwrite("data/LUT_elder_80.png", LUT_elder80);
	//-----------------------------
	//	作成時のガンマ特性をXML形式で保存
	//-----------------------------
	cout << "XML形式で保存しています..." << endl;
	FileStorage fs("data/LUT_elder.xml", FileStorage::WRITE);
	fs << "calibration" << "{"
		<< "display_gamma" << "{"
			<< "gamma_b" << gamma[0]
			<< "gamma_g" << gamma[1]
			<< "gamma_r" << gamma[2]
			<< "}"
		<< "display_luminance" << "{"
			<< "l_b" << Lmax[0]
			<< "l_g" << Lmax[1]
			<< "l_r" << Lmax[2]
			<< "}"
		<< "display_xy" << "{"
			<< "xy_b" << xyDisp[0]
			<< "xy_g" << xyDisp[1]
			<< "xy_r" << xyDisp[2]
			<< "}"
		<< "white_point" << "{"
			<< "xyY" << xyYWhite
			<< "}"
		<< "}";
	cout << "LUTの保存に成功しました．" << endl;

	return 0;
}

//	Point3f = Mat * Point3f　の行列演算オペレータ
Point3d operator*(Mat M, const Point3d& p)
{
	Mat src(3/*rows*/, 1 /* cols */, CV_64F);

	src.at<double>(0, 0) = p.x;
	src.at<double>(1, 0) = p.y;
	src.at<double>(2, 0) = p.z;

	Mat dst = M*src; //USE MATRIX ALGEBRA
	return Point3d(dst.at<double>(0, 0), dst.at<double>(1, 0), dst.at<double>(2, 0));
}
//	xyY to XYZ　の変換
Point3d cvtxyY2XYZ(Point3d xyY)
{
	return Point3d(
		xyY.z * xyY.x / xyY.y,
		xyY.z,
		xyY.z * (1 - xyY.x - xyY.y) / xyY.y);
}
Point3d cvtxyY2XYZ(Point2d xy, double Y)
{
	return cvtxyY2XYZ(Point3d(xy.x, xy.y, Y));
}
//	XYZ to xyY　の変換
Point3d cvtXYZ2xyY(Point3d XYZ)
{
	double sum = XYZ.x + XYZ.y + XYZ.z;
	return Point3d(XYZ.x / sum, XYZ.y / sum, XYZ.y);
}
//	XYZ to Lab　の変換
Point3d cvtXYZ2Lab(Point3d XYZ, Point3d XYZ0)
{
	Point3d Lab;
	if (XYZ.y / XYZ0.y > pow(6.0 / 29.0, 3))
	{
		Lab = Point3d(
			116.0 * pow(XYZ.y / XYZ0.y, 1.0 / 3.0) - 16.0,
			500.0 * (pow(XYZ.x / XYZ0.x, 1.0 / 3.0) - pow(XYZ.y / XYZ0.y, 1.0 / 3.0)),
			200.0 * (pow(XYZ.y / XYZ0.y, 1.0 / 3.0) - pow(XYZ.z / XYZ0.z, 1.0 / 3.0))
			);
	}
	else
	{
		Lab = Point3d(
			903.29 * XYZ.y / XYZ0.y,
			500.0 * (pow(XYZ.x / XYZ0.x, 1.0 / 3.0) - pow(XYZ.y / XYZ0.y, 1.0 / 3.0)),
			200.0 * (pow(XYZ.y / XYZ0.y, 1.0 / 3.0) - pow(XYZ.z / XYZ0.z, 1.0 / 3.0))
			);
	}
	return Lab;
}
//	Lab to XYZ　の変換
Point3d cvtLab2XYZ(Point3d Lab, Point3d XYZ0)
{
	return Point3d(
		pow(Lab.y / 500.0 + (Lab.x + 16.0) / 116.0, 3.0) * XYZ0.x,
		pow((Lab.x + 16.0) / 116.0, 3.0) * XYZ0.y,
		pow(-Lab.z / 200.0 + (Lab.x + 16.0) / 116.0, 3.0) * XYZ0.z
		);
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