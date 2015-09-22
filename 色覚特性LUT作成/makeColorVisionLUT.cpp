/***********************************************************************
	2�F�o�V�~�����[�V�����pLUT�쐬�v���O���� ver. 0.1
	Author: Nishizawa	Date: 2015/08/22
	�K�v�ȃ��C�u����: OpenCV 3.0

	�y�@�\�z
	2�F�o�҂̌�����_�˂̎�@�ōČ�����LUT���C
	PNG�`���CXML�`���̗����ō쐬���܂��D
	�A��XML�`���͔��ɏd���̂�PNG�`���𐄏����܂��D

	�y�Q�l�����z
	�_�ˏC�_�D����Ҏ��o�ƐF�o�ُ�̓���V�~�����[�V�����D

	�y���ӎ����z
	�{�v���O������OpenCV�̋@�\�ɏ����邽�߁C
	�S�Ă�RGB�֘A��BGR���ŋL�q���Ă���܂��D
	�i���̑��̐F��Ԃ̏��Ԃ͂��̂܂܂ł��j
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
void lookup(Mat &src, Mat &dst, Mat &LUT);

//----------------------------------------------
//	�J���[�L�����u���[�V�����v���l
//	�f�B�X�v���C�̃K���}�������������ɓ��͂���
//----------------------------------------------
static double gamma[3] = { 2.43434386, 1.96794267006194, 1.762290873 };		//	B, G, R�̃K���}�W��
static double Lmax[3] = { 31.23, 452.13, 150.13 };						//	B, G, R�̍ő�o�͋P�x(�K���l255�̎��̋P�x)
//	�f�B�X�v���C�̊e�u���̂̕��ϐF�x
//	xyDisp[0].x = xB�@�̂悤�ɂȂ�
static Point2d xyDisp[3] = {
	Point2d(0.1375, 0.04131),	//	xB, yB
	Point2d(0.3301, 0.6022),	//	xG, yG
	Point2d(0.6232, 0.3787)		//	xR, yR
};

//---------------------------------
//	��F�o�V�~�����[�V�����p�萔
//	�_�ˏC�_�E�v���O����������p
//---------------------------------
//	�����F�ƔF������P�F�h��
//	��^��F�o�C��^��F�o
static Point2d xy575nm(0.47878, 0.5202);
static Point2d xy475nm(0.1096, 0.08684);
//	�O�^��F�o
static Point2d xy660nm(0.72997, 0.27003);
static Point2d xy485nm(0.06871, 0.20072);
//	���G�l���M�[���F�_
static Point2d xyWhite(1.0 / 3.0, 1.0 / 3.0);
//	�����F���S(co-punctual point)
static Point2d confPoint_typeP(0.747, 0.253);		//	��^
static Point2d confPoint_typeD(1.080, -0.080);		//	��^
static Point2d confPoint_typeT(0.171, 0.000);		//	�O�^
//	YBGR to XYZ�̂��߂̍s��@xyDisp[3]����v�Z����
static Mat cvtMatYBGR2XYZ = (Mat_<double>(3, 3) <<
	xyDisp[0].x / xyDisp[0].y, xyDisp[1].x / xyDisp[1].y, xyDisp[2].x / xyDisp[2].y,
	1.0, 1.0, 1.0,
	(1 - xyDisp[0].x - xyDisp[0].y) / xyDisp[0].y, (1 - xyDisp[1].x - xyDisp[1].y) / xyDisp[1].y, (1 - xyDisp[2].x - xyDisp[2].y) / xyDisp[2].y);
//	��L�̋t�s��
static Mat cvtMatXYZ2YBGR = cvtMatYBGR2XYZ.inv();
//	Hunt-Pointer�ϊ�
//	�F�ϊ��̂��߂̍s��@CIE1931XYZ -> LMS
static Mat cvtMatXYZ2LMS = (Mat_<double>(3, 3) <<
	0.3982, 0.7040, -0.0804,
	-0.2268, 1.1679, 0.0458,
	0.0000, 0.0000, 0.8458);
//	��L�̋t�s��@LMS -> CIe1931ZXYZ
static Mat cvtMatLMS2XYZ = cvtMatXYZ2LMS.inv();
//	Smith-Pokorny�ϊ�
//	�F�ϊ��̂��߂̍s��@JuddXYZ -> LMS
static Mat cvtMatJuddXYZ2LMS = (Mat_<double>(3, 3) <<
	0.15514, 0.54312, -0.03286,
	-0.15514, 0.45684, 0.03286,
	0.0000, 0.0000, 0.01608);
//	��L�̋t�s��@LMS -> XYZ
static Mat cvtMatLMS2JuddXYZ = cvtMatJuddXYZ2LMS.inv();
//	CIE1931XYZ -> Judd�̏C�����F�֐�X'Y'Z'
static Mat cvtxy2Juddxy = (Mat_<double>(3, 3) <<
	1.0271, -0.00008, -0.00009,
	0.00376, 1.0072, 0.00764,
	0.08345, 0.01496, 1.0);

//-----------------------------
//	�e��萔�̌v�Z
//-----------------------------
//	xyY -> XYZ -> LMS
//	LMS.x = L, LMS.y = M, LMS.z = S
//	LMS�ȊO�͒P�ʃx�N�g���ŏ[���Ȃ̂�Y=1.0�ɌŒ�
static Point3d LMSWhite = cvtMatXYZ2LMS * cvtxyY2XYZ(xyWhite, 1.0);
static Point3d LMS575nm = cvtMatXYZ2LMS * cvtxyY2XYZ(xy575nm, 1.0);
static Point3d LMS475nm = cvtMatXYZ2LMS * cvtxyY2XYZ(xy475nm, 1.0);
static Point3d LMS660nm = cvtMatXYZ2LMS * cvtxyY2XYZ(xy660nm, 1.0);
static Point3d LMS485nm = cvtMatXYZ2LMS * cvtxyY2XYZ(xy485nm, 1.0);
static Point3d npd1 = LMSWhite.cross(LMS575nm);		//	575nm, White�Œ��铊�e���ʂ̖@���x�N�g��(P, D)
static Point3d npd2 = LMSWhite.cross(LMS475nm);		//	475nm, White�Œ��铊�e���ʂ̖@���x�N�g��(P, D)
static Point3d nt1 = LMSWhite.cross(LMS660nm);			//	660nm, White�Œ��铊�e���ʂ̖@���x�N�g��(T)
static Point3d nt2 = LMSWhite.cross(LMS485nm);			//	485nm, White�Œ��铊�e���ʂ̖@���x�N�g��(T)
static Point3d Ma1(
	cvtMatLMS2XYZ.at<double>(1, 0),
	cvtMatLMS2XYZ.at<double>(1, 1),
	cvtMatLMS2XYZ.at<double>(1, 2));	//	LMS -> XYZ�ϊ���Y�����@alpha, beta, gamma
//	�ϊ��s��̌v�Z
static Mat cvtMatYBGR2LMS = cvtMatXYZ2LMS * cvtMatYBGR2XYZ;	//	= cvtMatXYZ2LMS * cvtMatYBGR2XYZ
static Mat cvtMatLMS2YBGR = cvtMatXYZ2YBGR * cvtMatLMS2XYZ;	//	= cvtMatXYZ2YBGR * cvtMatLMS2XYZ

int main(void)
{
	//-------------------------------------------------
	//	���b�N�A�b�v�e�[�u���̍쐬
	//	b<<16 + g<<8 + r�@�̃}�N���ŃA�N�Z�X����
	//-------------------------------------------------
	Mat LUT_typeP(0xffffff + 1, 1, CV_8UC3);			//	��^(P)��F�o�V�~�����[�V�����p
	Mat LUT_typeD(0xffffff + 1, 1, CV_8UC3);			//	��^(D)��F�o�V�~�����[�V�����p
	Mat LUT_typeT(0xffffff + 1, 1, CV_8UC3);			//	�O�^(T)��F�o�V�~�����[�V�����p
	cout << "��F�oLUT���쐬��..." << endl;

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
					//	1. �K���}�����ƂɊK���l����\���f�o�C�X�o�͋P�x�����߂�
					//	YBGR.x = YG, YBGR.y = YG, YBGR.z = YR
					Point3d YBGR(	//	RGB�u���̂̏o�͋P�x
						Lmax[0] * pow((double)b / 255.0, gamma[0]),		//	B
						Lmax[1] * pow((double)g / 255.0, gamma[1]),		//	G
						Lmax[2] * pow((double)r / 255.0, gamma[2])		//	R
						);
					//	2. YBGR -> XYZ -> LMS
					Point3d LMS = cvtMatYBGR2LMS * YBGR;
					//	3. LMS��Ԃ�Brettel�V�~�����[�V����&�P�x�팸
					Point3d LMSBrettel_typeP, LMSBrettel_typeD, LMSBrettel_typeT;
					//	Type P
					//	L���̌���
					LMSBrettel_typeP = LMS;
					if (LMS.z / LMS.y < LMSWhite.z / LMSWhite.y)		//	MS���ʂ֐��ˉe�����F�x�N�g����White���X�����傫�����475nm�̕��ɓ��e
						LMSBrettel_typeP.x = -(npd1.y*LMS.y + npd1.z * LMS.z) / npd1.x;		//	�@���x�N�g���Œ�`����镽�ʂ�L�����s�ړ�
					else
						LMSBrettel_typeP.x = -(npd2.y*LMS.y + npd2.z * LMS.z) / npd2.x;
					double kp = 1.0 - Ma1.x * LMS.x / Ma1.dot(LMSBrettel_typeP);			//	L���̂���^����P�x�������J�b�g
					Point3d LMSKanbe_typeP = kp * LMSBrettel_typeP;
					//	Type D
					//	M���̌���
					LMSBrettel_typeD = LMS;
					if (LMS.z / LMS.x < LMSWhite.z / LMSWhite.x)
						LMSBrettel_typeD.y = -(npd1.z*LMS.z + npd1.x * LMS.x) / npd1.y;
					else
						LMSBrettel_typeD.y = -(npd2.z*LMS.z + npd2.x * LMS.x) / npd2.y;
					double kd = 1.0 - Ma1.y * LMS.y / Ma1.dot(LMSBrettel_typeD);
					Point3d LMSKanbe_typeD = kd * LMSBrettel_typeD;
					//	Type T
					//	S���̌���
					LMSBrettel_typeT = LMS;
					if (LMS.y / LMS.x < LMSWhite.y / LMSWhite.x)
						LMSBrettel_typeT.z = -(nt1.x*LMS.x + nt1.y * LMS.y) / nt1.z;
					else
						LMSBrettel_typeT.z = -(nt2.x*LMS.x + nt2.y * LMS.y) / nt2.z;
					double kt = 1.0 - Ma1.z * LMS.z / Ma1.dot(LMSBrettel_typeT);		//	���� kt = 1
					Point3d LMSKanbe_typeT = kt* LMSBrettel_typeT;
					//	4. LMS -> XYZ -> YBGR
					Vec3d YBGR_typeP = cvtMatLMS2YBGR * LMSKanbe_typeP;
					Vec3d YBGR_typeD = cvtMatLMS2YBGR * LMSKanbe_typeD;
					Vec3d YBGR_typeT = cvtMatLMS2YBGR * LMSKanbe_typeT;
					//	5. YBGR -> BGR(�K���}�␳)
					for (int i = 0; i < 3; i++)
					{
						double BGRp = 255.0 * pow(YBGR_typeP[i] / Lmax[i], 1 / gamma[i]);
						double BGRd = 255.0 * pow(YBGR_typeD[i] / Lmax[i], 1 / gamma[i]);
						double BGRt = 255.0 * pow(YBGR_typeT[i] / Lmax[i], 1 / gamma[i]);
						LUT_typeP.at<Vec3b>(BIT(b, g, r), 0)[i] = (BGRp > 0.0) ? ((BGRp < 255.0) ? (uchar)BGRp : 0xff) : 0x00;	//	0x00 ~ 0xff�@�͈̔͂Ɏ��߂�
						LUT_typeD.at<Vec3b>(BIT(b, g, r), 0)[i] = (BGRd > 0.0) ? ((BGRd < 255.0) ? (uchar)BGRd : 0xff) : 0x00;
						LUT_typeT.at<Vec3b>(BIT(b, g, r), 0)[i] = (BGRt > 0.0) ? ((BGRt < 255.0) ? (uchar)BGRt : 0xff) : 0x00;
					}
				}
			}
		}	//	LUT�쐬�I��
	}
	cout << "\nLUT�̍쐬���I�����܂����D�e�X�g�摜��\�����܂��D" << endl;
	//-----------------------------
	//	�摜�ϊ��e�X�g
	//-----------------------------
	Mat testImg = imread("img/Baboon.png");
	Mat dstImgP, dstImgD, dstImgT;
	lookup(testImg, dstImgP, LUT_typeP);
	lookup(testImg, dstImgD, LUT_typeD);
	lookup(testImg, dstImgT, LUT_typeT);
	imshow("�I���W�i��", testImg);
	imshow("P�^��F�o", dstImgP);
	imshow("D�^��F�o", dstImgD);
	imshow("T�^��F�o", dstImgT);
	waitKey(0);
	destroyAllWindows();

	cout << "���ʂ�ۑ���..." << endl;
	imwrite("img/Baboon1.png", dstImgP);
	imwrite("img/Baboon2.png", dstImgD);
	imwrite("img/Baboon3.png", dstImgT);
	//-----------------------------
	//	LUT��PNG�`���ŕۑ�
	//-----------------------------
	imwrite("data/LUT_dichromat_typeP.bmp", LUT_typeP);
	imwrite("data/LUT_dichromat_typeD.bmp", LUT_typeD);
	imwrite("data/LUT_dichromat_typeT.bmp", LUT_typeT);
	//-----------------------------
	//	LUT��XML�`���ŕۑ�
	//-----------------------------
	cout << "XML�`���ŕۑ����Ă��܂�..." << endl;
	FileStorage fs("data/LUT_dichromat.xml", FileStorage::WRITE);
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
		<< "type_p" << LUT_typeP
		<< "type_d" << LUT_typeD
		<< "type_t" << LUT_typeT
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