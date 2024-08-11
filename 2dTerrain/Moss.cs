using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace TerrainGenerator
{
    public unsafe class Moss
    {
        public byte* image;
        public int width;
        public int height;
        public Moss(BitmapData bitmapData)
        {
            image = (byte*)bitmapData.Scan0;
            width = bitmapData.Width;
            height = bitmapData.Height;
        }
        [DllImport("noise2d.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern float* GeneratePerlinNoise(int width, int height, int octaveCount);

        public void OverlayMoss(double density, int noisedepth)
        {
            var perlin = GeneratePerlinNoise(width, height, noisedepth);
            string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            var mossimage = (Bitmap)Image.FromFile(exePath + "\\images\\mossseam.png");
            BitmapData mossdata = mossimage.LockBits(new Rectangle(0, 0, mossimage.Width, mossimage.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppPArgb);
            byte* moss_scan0 = (byte*)mossdata.Scan0;
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    //Apply moss based off of perlin noise
                    double blendfactor = perlin[x + y * width] * density;
                    Extensions.BlendColors(image + x * 4 + y * width * 4, moss_scan0 + ((x % mossdata.Width) * 4) + ((y % mossdata.Height) * 4 * mossdata.Width), blendfactor);
                }
            }
            mossimage.UnlockBits(mossdata);
        }
    }
}