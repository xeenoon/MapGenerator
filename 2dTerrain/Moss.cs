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
            float* perlin;
            if (Extensions.HasNvidiaGpu())
            {
                perlin = GeneratePerlinNoise(width, height, noisedepth);
            }
            else
            {
                perlin = JaggedArrayToFloatPointer(PerlinNoise.GeneratePerlinNoise(width, height, noisedepth), width * height);
            }

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
            if (Extensions.HasNvidiaGpu()) //Dont free if we dont have to
            {
                Marshal.FreeHGlobal((IntPtr)perlin);
            }
        }
        public static float* JaggedArrayToFloatPointer(float[][] jaggedArray, int length)
        {
            // Determine the total length of the flattened array
            int rows = jaggedArray.Length;
            int cols = jaggedArray.Length > 0 ? jaggedArray[0].Length : 0;

            // Flatten the jagged array into a single-dimensional array
            float[] flatArray = new float[length];

            for (int i = 0; i < rows; i++)
            {
                Array.Copy(jaggedArray[i], 0, flatArray, i * cols, cols);
            }

            // Pin the array in memory
            GCHandle handle = GCHandle.Alloc(flatArray, GCHandleType.Pinned);
            float* pointer = (float*)handle.AddrOfPinnedObject();

            return pointer;
        }
    }
}