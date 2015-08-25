/***********************************************************************
	����ғ����V�~�����[�V�����pLUT�쐬�v���O���� ver. 0.1
	Author: Nishizawa	Date: 2015/08/22
	�K�v�ȃ��C�u����: OpenCV 3.0
	
	�y�@�\�z
	����҂̌�����_�˂̎�@�ōČ�����LUT���C
	PNG�`���CXML�`���̗����ō쐬���܂��D
	�A��XML�`���͔��ɏd���̂�PNG�`���𐄏����܂��D

	�y�Q�l�����z
	�_�ˏC�_�D����Ҏ��o�ƐF�o�ُ�̓���V�~�����[�V�����D
	���c�C�_�D����҂̐F�o�������l���������F���̕]�����@�Ɖ����D
	�⌳�C�_�D���o��Q�V�~�����[�^�ɂ������Ԏ��g�������͋[�̐��k���D

	�y���ӎ����z
	�{�v���O������OpenCV�̋@�\�ɏ����邽�߁C
	�S�Ă�RGB�֘A��BGR���ŋL�q���Ă���܂��D
	�i���̑��̐F��Ԃ͂��̂܂܂ł��j
	���ׁ̈CXYZ - RGB�ϊ����ōs�񂪂����ƈႢ�܂��̂�
	���ӂ��ēǂ�ł��������D
************************************************************************/

#include "OpenCV3Linker.h"

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
void lookup(Mat &src, Mat &dst, Mat &LUT);

//----------------------------------------------
//	�J���[�L�����u���[�V�����v���l
//	�f�B�X�v���C�̃K���}�������������ɓ��͂���
//----------------------------------------------
static double gamma[3] = { 2.28905804, 2.493837294, 2.232554852 };		//	B, G, R�̃K���}�W��
static double Lmax[3] = { 51.28, 223.98, 68.48 };						//	B, G, R�̍ő�o�͋P�x(�K���l255�̎��̋P�x)
//	�f�B�X�v���C�̊e�u���̂̕��ϐF�x
//	xyDisp[0].x = xB�@�̂悤�ɂȂ�
static Point2d xyDisp[3] = {
	Point2d(0.1751, 0.09136),	//	xB, yB
	Point2d(0.2934, 0.6378),	//	xG, yG
	Point2d(0.6311, 0.3539)		//	xR, yR
};

//---------------------------------
//	����҃V�~�����[�V�����p�萔
//	�_�ˁC���c�̏C�_������p
//---------------------------------
static Point3d K70(0.521, 0.673, 0.864);		//	70�� vs 20�΂̎����P�x��CBGR
static Point3d K80(0.388, 0.555, 0.847);		//	80�� vs 20��, BGR
static double alpha70 = 0.6, alpha80 = 0.5;			//	�V�l���k���ɂ��Ԗ��Ɠx�ቺ��\���������ߗ�

int main(void)
{
	//-------------------------------------------------
	//	���b�N�A�b�v�e�[�u���̍쐬
	//	b<<16 + g<<8 + r�@�̃}�N���ŃA�N�Z�X����
	//-------------------------------------------------
	Mat LUT_elder70(0xffffff + 1, 1, CV_8UC3);			//	70�΍���҃V�~�����[�V�����p
	Mat LUT_elder80(0xffffff + 1, 1, CV_8UC3);			//	80�΍���҃V�~�����[�V�����p
	cout << "����ғ���LUT���쐬��..." << endl;
	for (int b = 0; b <= 0xff; b++)
	{
		static int counter = 1;
		cout << counter++ << " / 256 ����" << "\r";
		for (int g = 0; g <= 0xff; g++)
		{
			for (int r = 0; r <= 0xff; r++)
			{
				//	1. �K���}�����ƂɊK���l����\���f�o�C�X�o�͋P�x�����߂�
				//	YBGR.x = YG, YBGR.y = YG, YBGR.z = YR
				Point3d YBGR(	//	RGB�u���̂̏o�͋P�x
					Lmax[0] * pow((double)b / 0xff, gamma[0]),		//	B
					Lmax[1] * pow((double)g / 0xff, gamma[1]),		//	G
					Lmax[2] * pow((double)r / 0xff, gamma[2])		//	R
					);
				//	2. �V�l���k�����l�����������P�x��̌v�Z
				double Ke70 = alpha70 * (K70.x * YBGR.x + K70.y * YBGR.y + K70.z * YBGR.z) / (YBGR.x + YBGR.y + YBGR.z);
				double Ke80 = alpha80 * (K80.x * YBGR.x + K80.y * YBGR.y + K80.z * YBGR.z) / (YBGR.x + YBGR.y + YBGR.z);
				//	3. �����P�x�̌v�Z
				//	Ye70.x = YeB(70�Εϊ����B�u���̂̏o�͋P�x), Ye70.y = YeG, Ye70.z = YeR
				Vec3d Ye70 = Ke70 * YBGR;
				Vec3d Ye80 = Ke80 * YBGR;
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
	cout << "\nLUT�̍쐬���I�����܂����D�e�X�g�摜��\�����܂��D" << endl;
	//-----------------------------
	//	�摜�ϊ��e�X�g
	//-----------------------------
	Mat testImg = imread("img/test.bmp");
	Mat dstImg70, dstImg80;
	lookup(testImg, dstImg70, LUT_elder70);
	lookup(testImg, dstImg80, LUT_elder80);
	imshow("�I���W�i��", testImg);
	imshow("70�Εϊ�", dstImg70);
	imshow("80�Εϊ�", dstImg80);
	waitKey(0);
	destroyAllWindows();

	cout << "���ʂ�ۑ���..." << endl;
	imwrite("img/test70.bmp", dstImg70);
	imwrite("img/test80.bmp", dstImg80);
	//-----------------------------
	//	LUT��PNG�`���ŕۑ�
	//-----------------------------
	imwrite("data/LUT_elder_70.png", LUT_elder70);
	imwrite("data/LUT_elder_80.png", LUT_elder80);
	//-----------------------------
	//	LUT��XML�`���ŕۑ�
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
		<< "}";
	fs << "LUT" << "{"
		<< "seventy" << LUT_elder70
		<< "eighty" << LUT_elder80
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
	return Point3d(xyY.z * xyY.x / xyY.y, xyY.z, xyY.z * (1 - xyY.x - xyY.y) / xyY.y);
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