using System.Drawing.Imaging;

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
        public void OverlayMoss(int density, int noisedepth)
        {
            var noise = PerlinNoise.GenerateWhiteNoise(width, height);
            var perlin = PerlinNoise.GeneratePerlinNoise(width, height, 4);
            var mossimage = (Bitmap)Image.FromFile("C:\\Users\\ccw10\\Downloads\\mossseam.png");
            BitmapData mossdata = mossimage.LockBits(new Rectangle(0,0,mossimage.Width,mossimage.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppPArgb);
            byte* moss_scan0 = (byte*)mossdata.Scan0;
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    //Apply moss based off of perlin noise
                    double blendfactor = perlin[x][y];
                    BlendColors(image+x*4 + y*height*4, moss_scan0 + ((x*4) % mossdata.Width) + ((y * mossdata.Stride) % mossdata.Width), blendfactor);
                }
            }
            mossimage.UnlockBits(mossdata);
        }
        public static void BlendColors(byte* from, byte* to, double blendfactor)
        {
            //Start from the color 0 blend towards color 1
            int r_difference = to[2]-from[2];
            from[2] += (byte)(r_difference*blendfactor); //Blend factor of 0 will be original color, blend factor of 1 will e completely the next color
            int g_difference = to[1]-from[1];
            from[1] += (byte)(g_difference*blendfactor); //Blend factor of 0 will be original color, blend factor of 1 will e completely the next color
            int b_difference = to[0]-from[0];
            from[0] += (byte)(b_difference*blendfactor); //Blend factor of 0 will be original color, blend factor of 1 will e completely the next color
        }
    }
}