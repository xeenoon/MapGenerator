using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TerrainGenerator
{
    public unsafe class NormalMap
    {
        public NormalMap(Bitmap normalmap, Bitmap originalimage)
        {
            this.normalmap = normalmap;
            this.originalimage = originalimage;
        }
        public Bitmap normalmap;
        public Bitmap originalimage;
        [DllImport("vectorexample.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void ApplyNormalMap(byte* original, byte* normal, byte* output, int width, int height);
        public void ApplyNormalMap()
        {
            Bitmap readbitmap = (Bitmap)originalimage.Clone();
            var readdata = readbitmap.LockBits(new Rectangle(0, 0, originalimage.Width, originalimage.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            var readdataptr = (byte*)readdata.Scan0;

            var writedata = originalimage.LockBits(new Rectangle(0, 0, originalimage.Width, originalimage.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            var writedataptr = (byte*)writedata.Scan0;

            var normaldata = normalmap.LockBits(new Rectangle(0, 0, normalmap.Width, normalmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            var normalptr = (byte*)normaldata.Scan0;

            ApplyNormalMap(readdataptr, normalptr, writedataptr, readdata.Width, readdata.Height);

            originalimage.UnlockBits(writedata);
            normalmap.UnlockBits(normaldata);
            readbitmap.UnlockBits(readdata);
        }
        private void ApplyNormalMap(byte* originalptr, byte* normalptr)
        {
            for (int y = 0; y < originalimage.Height; y++)
            {
                for (int x = 0; x < originalimage.Width; x++)
                {
                    byte* originalcolor = originalptr + (x * 4 + y * 4 * originalimage.Width);
                    byte* normalcolor = normalptr + (x * 4 + y * 4 * originalimage.Width);

                    // Convert normal color to a vector
                    float nx = (normalcolor[2] / 255.0f) * 2.0f - 1.0f;
                    float ny = (normalcolor[1] / 255.0f) * 2.0f - 1.0f;
                    float nz = (normalcolor[0] / 255.0f) * 2.0f - 1.0f;

                    // Simple light direction (from top-left)
                    float lx = 0.5f;
                    float ly = -0.5f;
                    float lz = 1.0f;

                    // Normalize the light direction
                    float length = (float)Math.Sqrt(lx * lx + ly * ly + lz * lz);
                    lx /= length;
                    ly /= length;
                    lz /= length;

                    // Compute the dot product of the normal and light direction
                    float dot = nx * lx + ny * ly + nz * lz;
                    dot = Math.Max(0.0f, dot); // Clamp to [0, 1]

                    // Apply the dot product to the original color to get the shaded color
                    byte r = (byte)(originalcolor[2] * dot);
                    byte g = (byte)(originalcolor[1] * dot);
                    byte b = (byte)(originalcolor[0] * dot);
                    originalcolor[2] = r;
                    originalcolor[1] = g;
                    originalcolor[0] = b;
                }
            }
        }
        public static Bitmap GenerateNormalMap(Bitmap bitmap, float intensity)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            Bitmap normalMap = new Bitmap(width, height, PixelFormat.Format24bppRgb);

            // Lock bits for both the input bitmap and the output normal map
            var blurredbitmap = ApplyGaussianBlur(bitmap, 0.5f);

            BitmapData bitmapData = blurredbitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            BitmapData normalMapData = normalMap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            int stride = bitmapData.Stride;
            IntPtr bitmapPtr = bitmapData.Scan0;
            IntPtr normalMapPtr = normalMapData.Scan0;

            byte* bitmapPixels = (byte*)bitmapPtr;
            byte* normalMapPixels = (byte*)normalMapPtr;

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    // Calculate the index in the pixel array
                    int index = y * stride + x * 3;

                    // Get the color intensity of the surrounding pixels
                    float tl = bitmapPixels[index - stride - 3];
                    float t = bitmapPixels[index - stride];
                    float tr = bitmapPixels[index - stride + 3];
                    float l = bitmapPixels[index - 3];
                    float r = bitmapPixels[index + 3];
                    float bl = bitmapPixels[index + stride - 3];
                    float b = bitmapPixels[index + stride];
                    float br = bitmapPixels[index + stride + 3];

                    // Calculate the gradients
                    float dx = (tr + 2 * r + br) - (tl + 2 * l + bl);
                    float dy = (bl + 2 * b + br) - (tl + 2 * t + tr);

                    dx *= intensity;
                    dy *= intensity;

                    // Calculate the normal vector
                    float dz = 255.0f / 2.0f;
                    float length = (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
                    float nx = dx / length;
                    float ny = dy / length;
                    float nz = dz / length;

                    // Convert normal vector to color
                    byte rNormal = (byte)((nx + 1) * 127.5f);
                    byte gNormal = (byte)((ny + 1) * 127.5f);
                    byte bNormal = (byte)((nz + 1) * 127.5f);

                    // Set the pixel in the normal map
                    normalMapPixels[index] = bNormal;
                    normalMapPixels[index + 1] = gNormal;
                    normalMapPixels[index + 2] = rNormal;
                }
            }


            // Unlock the bits
            blurredbitmap.UnlockBits(bitmapData);
            normalMap.UnlockBits(normalMapData);

            return normalMap;
        }
        public static Bitmap GenerateNormalMap(Bitmap bitmap, float intensity, Puddle puddle)
        {

            Bitmap polygonmarker = new Bitmap(bitmap.Width, bitmap.Height);
            Graphics graphics = Graphics.FromImage(polygonmarker);
            graphics.FillPolygon(new Pen(Color.FromArgb(255, 255, 0, 0)).Brush, puddle.bounds.ToArray()); //Mark the pixel

            var markdata = polygonmarker.LockBits(new Rectangle(0, 0, polygonmarker.Width, polygonmarker.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);
            var markptr = (byte*)markdata.Scan0;

            int width = bitmap.Width;
            int height = bitmap.Height;
            Bitmap normalMap = new Bitmap(width, height, PixelFormat.Format24bppRgb);

            // Lock bits for both the input bitmap and the output normal map
            var blurredbitmap = ApplyGaussianBlur(bitmap, 0.5f);

            BitmapData bitmapData = blurredbitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            BitmapData normalMapData = normalMap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            int stride = bitmapData.Stride;
            IntPtr bitmapPtr = bitmapData.Scan0;
            IntPtr normalMapPtr = normalMapData.Scan0;

            byte* bitmapPixels = (byte*)bitmapPtr;
            byte* normalMapPixels = (byte*)normalMapPtr;

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {

                    // Calculate the index in the pixel array
                    int index = y * stride + x * 3;

                    // Get the color intensity of the surrounding pixels
                    float tl = bitmapPixels[index - stride - 3];
                    float t = bitmapPixels[index - stride];
                    float tr = bitmapPixels[index - stride + 3];
                    float l = bitmapPixels[index - 3];
                    float r = bitmapPixels[index + 3];
                    float bl = bitmapPixels[index + stride - 3];
                    float b = bitmapPixels[index + stride];
                    float br = bitmapPixels[index + stride + 3];

                    // Calculate the gradients
                    float dx = (tr + 2 * r + br) - (tl + 2 * l + bl);
                    float dy = (bl + 2 * b + br) - (tl + 2 * t + tr);

                    dx *= intensity;
                    dy *= intensity;

                    // Calculate the normal vector
                    float dz = 255.0f / 2.0f;
                    float length = (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
                    float nx = dx / length;
                    float ny = dy / length;
                    float nz = dz / length;

                    // Convert normal vector to color
                    byte rNormal = (byte)((nx + 1) * 127.5f);
                    byte gNormal = (byte)((ny + 1) * 127.5f);
                    byte bNormal = (byte)((nz + 1) * 127.5f);

                    // Set the pixel in the normal map
                    normalMapPixels[index] = bNormal;
                    normalMapPixels[index + 1] = gNormal;
                    normalMapPixels[index + 2] = rNormal;
                    if (markptr[x * 4 + y * 4 * bitmap.Width + 2] == 255) // Am I in the polygon?
                    {
                        const byte recessedval = 225;
                        const double blenddst = 200;
                        var distance = puddle.DistanceTo(new Point(x, y));
                        if (distance <= blenddst && distance != -1) //Blend on edges
                        {
                            double blendfactor = Math.Min(distance / blenddst, 1);

                            byte[] normalArray = { recessedval, recessedval, recessedval };
                            fixed (byte* newnormals = normalArray)
                            {
                                Extensions.BlendColors(normalMapPixels + index, newnormals, blendfactor);
                            }

                        }
                        else
                        {
                            // Set the normal vector to a recessed value (e.g., pointing inward)
                            normalMapPixels[index] = recessedval; // Arbitrary recessed value for blue
                            normalMapPixels[index + 1] = recessedval; // Arbitrary recessed value for green
                            normalMapPixels[index + 2] = recessedval;   // Arbitrary recessed value for red
                        }
                    }
                }
            }

            // Unlock the bits
            blurredbitmap.UnlockBits(bitmapData);
            normalMap.UnlockBits(normalMapData);

            return normalMap;
        }
        public static Bitmap ApplyGaussianBlur(Bitmap bitmap, float intensity)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            Bitmap blurredBitmap = new Bitmap(width, height);

            // Lock bits for the input bitmap and the output blurred bitmap
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            BitmapData blurredBitmapData = blurredBitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            int stride = bitmapData.Stride;
            IntPtr bitmapPtr = bitmapData.Scan0;
            IntPtr blurredBitmapPtr = blurredBitmapData.Scan0;

            byte* bitmapPixels = (byte*)bitmapPtr;
            byte* blurredPixels = (byte*)blurredBitmapPtr;

            // Gaussian kernel (3x3)
            float[,] kernel = {
                { 1/16f, 2/16f, 1/16f },
                { 2/16f, 4/16f, 2/16f },
                { 1/16f, 2/16f, 1/16f }
            }; //Sample more from the centre, less from the edges and even less from the corners
               //Sum is 16 so all kernel values add up to 1
               //Apply this percentage blur to all the nearby pixels nearby

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    float blue = 0.0f, green = 0.0f, red = 0.0f;

                    // Apply Gaussian blur kernel
                    for (int ky = -1; ky <= 1; ky++)
                    {
                        for (int kx = -1; kx <= 1; kx++)
                        {
                            int pixelIndex = ((y + ky) * stride) + ((x + kx) * 3);
                            blue += bitmapPixels[pixelIndex] * kernel[ky + 1, kx + 1];
                            green += bitmapPixels[pixelIndex + 1] * kernel[ky + 1, kx + 1];
                            red += bitmapPixels[pixelIndex + 2] * kernel[ky + 1, kx + 1];
                        }
                    }

                    // Apply intensity
                    blue = blue * intensity + bitmapPixels[y * stride + x * 3] * (1 - intensity);
                    green = green * intensity + bitmapPixels[y * stride + x * 3 + 1] * (1 - intensity);
                    red = red * intensity + bitmapPixels[y * stride + x * 3 + 2] * (1 - intensity);

                    // Set the blurred pixel
                    int index = y * stride + x * 3;
                    blurredPixels[index] = (byte)blue;
                    blurredPixels[index + 1] = (byte)green;
                    blurredPixels[index + 2] = (byte)red;
                }

            }

            // Unlock the bits
            bitmap.UnlockBits(bitmapData);
            blurredBitmap.UnlockBits(blurredBitmapData);

            return blurredBitmap;
        }
    }
}