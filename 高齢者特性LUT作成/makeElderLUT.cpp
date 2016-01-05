/***********************************************************************
	����ғ����V�~�����[�V�����pLUT�쐬�v���O���� ver. 0.2
	Author: Nishizawa	Date: 2016/01/05
	�K�v�ȃ��C�u����: OpenCV 3.0
	
	�y�@�\�z
	����҂̌��������c�E����̎�@�ōČ�����LUT���C
	PNG�`���CXML�`���̗����ō쐬���܂��D
	�A��XML�`���͔��ɏd���̂�PNG�`���𐄏����܂��D

	�y�Q�l�����z
	�_�ˏC�_�D����Ҏ��o�ƐF�o�ُ�̓���V�~�����[�V�����D
	���c�C�_�D����҂̐F�o�������l���������F���̕]�����@�Ɖ����D
	���㑲�_�D����҂̎��o�����V�~�����[�^�Ɋւ��錤���D
	�⌳�C�_�D���o��Q�V�~�����[�^�ɂ������Ԏ��g�������͋[�̐��k���D

	�y���ӎ����z
	�{�v���O������OpenCV�̋@�\�ɏ����邽�߁C
	�S�Ă�RGB�֘A��BGR���ŋL�q���Ă���܂��D
	�i���̑��̐F��Ԃ͂��̂܂܂ł��j
	���ׁ̈CXYZ - RGB�ϊ����ōs�񂪂����ƈႢ�܂��̂�
	���ӂ��ēǂ�ł��������D
************************************************************************/

#include "OpenCV3Linker.h"
#include <omp.h>

using namespace cv;
using namespace std;

//	LUT�̃A�N�Z�X�p�}�N��
#define BIT(B, G, R) (((int)(B) << 16) | ((int)(G) << 8) | (R))

//--------------------------------------------
//	�֐��v���g�^�C�v
//--------------------------------------------
Point3d operator*(Mat M, const Point3d& p);
Point3d cvtxyY2XYZ(Point3d xyY);
Point3d cvtxyY2XYZ(Point2d xy, double Y);
Point3d cvtXYZ2xyY(Point3d XYZ);
Point3d cvtXYZ2Lab(Point3d XYZ, Point3d XYZ0);
Point3d cvtLab2XYZ(Point3d Lab, Point3d XYZ0);
void lookup(Mat &src, Mat &dst, Mat &LUT);

//----------------------------------------------
//	�J���[�L�����u���[�V�����v���l
//	�f�B�X�v���C�̃K���}�������������ɓ��͂���
//----------------------------------------------
static double gamma[3] = { 2.43434386, 1.96794267, 1.762290873};		//	B, G, R�̃K���}�W��
static double Lmax[3] = { 31.23, 452.13, 150.13 };						//	B, G, R�̍ő�o�͋P�x(�K���l255�̎��̋P�x)
//	�f�B�X�v���C�̊e�u���̂̕��ϐF�x
//	xyDisp[0].x = xB�@�̂悤�ɂȂ�
static Point2d xyDisp[3] = {
	Point2d(0.1375, 0.04131),	//	xB, yB
	Point2d(0.3301, 0.6022),	//	xG, yG
	Point2d(0.6232, 0.3787)		//	xR, yR
};
static Point3d xyYWhite(0.318, 0.335, 631);		//	���F�i���S�g�U���˖ʁj�̑��F�l

//---------------------------------
//	����҃V�~�����[�V�����p�萔
//	�_�ˁC���c�̏C�_������p
//---------------------------------
static Point3d K70(0.521, 0.673, 0.864);		//	70�� vs 20�΂̎����P�x��CBGR
static Point3d K80(0.388, 0.555, 0.847);		//	80�� vs 20��, BGR
static double alpha70 = 0.6, alpha80 = 0.5;			//	�V�l���k���ɂ��Ԗ��Ɠx�ቺ��\���������ߗ�

//	YBGR to XYZ�̂��߂̍s��@xyDisp[3]����v�Z����
static Mat cvtMatYBGR2XYZ = (Mat_<double>(3, 3) <<
	xyDisp[0].x / xyDisp[0].y, xyDisp[1].x / xyDisp[1].y, xyDisp[2].x / xyDisp[2].y,
	1.0, 1.0, 1.0,
	(1 - xyDisp[0].x - xyDisp[0].y) / xyDisp[0].y, (1 - xyDisp[1].x - xyDisp[1].y) / xyDisp[1].y, (1 - xyDisp[2].x - xyDisp[2].y) / xyDisp[2].y);
//	��L�̋t�s��
static Mat cvtMatXYZ2YBGR = cvtMatYBGR2XYZ.inv();
//	���S�g�U���˖ʂ�XYZ
static Point3d XYZWhite = cvtxyY2XYZ(xyYWhite);

int main(void)
{
	//-------------------------------------------------
	//	���b�N�A�b�v�e�[�u���̍쐬
	//	b<<16 + g<<8 + r�@�̃}�N���ŃA�N�Z�X����
	//-------------------------------------------------
	Mat LUT_elder70(0xffffff + 1, 1, CV_8UC3);			//	70�΍���҃V�~�����[�V�����p
	Mat LUT_elder80(0xffffff + 1, 1, CV_8UC3);			//	80�΍���҃V�~�����[�V�����p
	cout << "����ғ���LUT���쐬��..." << endl;
#pragma omp parallel
	{
#pragma omp for
		for (int b = 0; b <= 0xff; b++)
		{
			static int counter = 1;
			cout << counter++ << " / 256 ����" << "\r";
			for (int g = 0; g <= 0xff; g++)
			{
				for (int r = 0; r <= 0xff; r++)
				{
					//	1. �K���}�����ƂɊK���lBGR����\���f�o�C�X�o�͋P�xYBGR�����߂�
					//	YBGR.x = YG, YBGR.y = YG, YBGR.z = YR
					Point3d YBGR(	//	RGB�u���̂̏o�͋P�x
						Lmax[0] * pow((double)b / 0xff, gamma[0]),		//	B
						Lmax[1] * pow((double)g / 0xff, gamma[1]),		//	G
						Lmax[2] * pow((double)r / 0xff, gamma[2])		//	R
						);
					//	2. �F�P�퐫���l�������F���E�ʓx�ϊ����̓����i���c�E����j
					//	���ӁF���̕ϊ����͕���24.4��vs����68.2�΂̎������ʂɊ�Â�
					//	2.1. YBGR -> XYZ -> L*a*b*
					Point3d XYZ = cvtMatYBGR2XYZ * YBGR;
					Point3d Lab = cvtXYZ2Lab(XYZ, XYZWhite);
					//	2.2. �F���E�ʓx�̕ϊ��W���𓱏o
					double chroma = atan2(Lab.y, Lab.x);		//	�ʓx
					double hue = sqrt(Lab.y * Lab.y + Lab.z * Lab.z);	//	�F��[rad]
					double Cey = 1.32 * exp(-chroma / 6.53) - exp(-pow((chroma - 47.23) / 2680.53, 2)) / 27.89 + 1.0;
					double dth = 0.131 * cos(2 * CV_PI * (hue - 5.875) / 5.686) + 0.027;
					//	2.3. ����ҐF�x�ϊ�
					Point3d Lab1(
						Lab.x,
						Cey * (cos(dth) * Lab.y - sin(dth) * Lab.z),
						Cey * (sin(dth) * Lab.y + cos(dth) * Lab.z)
						);
					//	2.4. L*a*b* -> XYZ -> YBGR
					Point3d XYZ1 = cvtLab2XYZ(Lab1, XYZWhite);
					Point3d YBGR1 = cvtMatXYZ2YBGR * XYZ;
					//	3. �V�l���k�����l�����������P�x��̌v�Z�i�_�ˁj
					double Ke70 = alpha70 * (K70.x * YBGR1.x + K70.y * YBGR1.y + K70.z * YBGR1.z) / (YBGR1.x + YBGR1.y + YBGR1.z);
					double Ke80 = alpha80 * (K80.x * YBGR1.x + K80.y * YBGR1.y + K80.z * YBGR1.z) / (YBGR1.x + YBGR1.y + YBGR1.z);
					//	4. �����P�x�̌v�Z
					//	Ye70.x = YeB(70�Εϊ����B�u���̂̏o�͋P�x), Ye70.y = YeG, Ye70.z = YeR
					Vec3d Ye70 = Ke70 * YBGR1;
					Vec3d Ye80 = Ke80 * YBGR1;
					//	4. �K���}�␳��BGR�ɋt�Z
					for (int i = 0; i < 3; i++)
					{
						double BGR70 = 0xff * pow(Ye70[i] / Lmax[i], 1.0 / gamma[i]);
						double BGR80 = 0xff * pow(Ye80[i] / Lmax[i], 1.0 / gamma[i]);

						LUT_elder70.at<Vec3b>(BIT(b, g, r), 0)[i] = (BGR70 > 0.0) ? ((BGR70 < 255.0) ? (uchar)BGR70 : 0xff) : 0x00;
						LUT_elder80.at<Vec3b>(BIT(b, g, r), 0)[i] = (BGR80 > 0.0) ? ((BGR80 < 255.0) ? (uchar)BGR80 : 0xff) : 0x00;
					}
				}
			}
		}		//	LUT�쐬�I��
	}
	cout << "\nLUT�̍쐬���I�����܂����D�e�X�g�摜��\�����܂��D" << endl;
	//-----------------------------
	//	�摜�ϊ��e�X�g
	//-----------------------------
	Mat testImg = imread("img/sapporo_new.png");
	Mat dstImg70, dstImg80;
	lookup(testImg, dstImg70, LUT_elder70);
	lookup(testImg, dstImg80, LUT_elder80);
	imshow("�I���W�i��", testImg);
	imshow("70�Εϊ�", dstImg70);
	imshow("80�Εϊ�", dstImg80);
	waitKey(0);
	destroyAllWindows();

	cout << "���ʂ�ۑ���..." << endl;
	imwrite("img/sapporo_new70.png", dstImg70);
	imwrite("img/sapporo_new80.png", dstImg80);
	//-----------------------------
	//	LUT��PNG�`���ŕۑ�
	//-----------------------------
	imwrite("data/LUT_elder_70.png", LUT_elder70);
	imwrite("data/LUT_elder_80.png", LUT_elder80);
	//-----------------------------
	//	�쐬���̃K���}������XML�`���ŕۑ�
	//-----------------------------
	cout << "XML�`���ŕۑ����Ă��܂�..." << endl;
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
	cout << "LUT�̕ۑ��ɐ������܂����D" << endl;

	return 0;
}

//	Point3f = Mat * Point3f�@�̍s�񉉎Z�I�y���[�^
Point3d operator*(Mat M, const Point3d& p)
{
	Mat src(3/*rows*/, 1 /* cols */, CV_64F);

	src.at<double>(0, 0) = p.x;
	src.at<double>(1, 0) = p.y;
	src.at<double>(2, 0) = p.z;

	Mat dst = M*src; //USE MATRIX ALGEBRA
	return Point3d(dst.at<double>(0, 0), dst.at<double>(1, 0), dst.at<double>(2, 0));
}
//	xyY to XYZ�@�̕ϊ�
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
//	XYZ to xyY�@�̕ϊ�
Point3d cvtXYZ2xyY(Point3d XYZ)
{
	double sum = XYZ.x + XYZ.y + XYZ.z;
	return Point3d(XYZ.x / sum, XYZ.y / sum, XYZ.y);
}
//	XYZ to Lab�@�̕ϊ�
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
//	Lab to XYZ�@�̕ϊ�
Point3d cvtLab2XYZ(Point3d Lab, Point3d XYZ0)
{
	return Point3d(
		pow(Lab.y / 500.0 + (Lab.x + 16.0) / 116.0, 3.0) * XYZ0.x,
		pow((Lab.x + 16.0) / 116.0, 3.0) * XYZ0.y,
		pow(-Lab.z / 200.0 + (Lab.x + 16.0) / 116.0, 3.0) * XYZ0.z
		);
}

//	LUT�ɏ]���ĉ�f�l�̓���ւ�
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