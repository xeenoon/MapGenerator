using System.Drawing.Imaging;

namespace TerrainGenerator
{
    public class NormalMap
    {
        public NormalMap(Bitmap normalmap, Bitmap originalimage)
        {
            this.normalmap = normalmap;
            this.originalimage = originalimage;
        }
        public Bitmap normalmap;
        public Bitmap originalimage;
        private unsafe void ApplyNormalMap()
        {
            var originaldata = originalimage.LockBits(new Rectangle(0, 0, originalimage.Width, originalimage.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            var originalptr = (byte*)originaldata.Scan0;
            var normaldata = normalmap.LockBits(new Rectangle(0, 0, normalmap.Width, normalmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            var normalptr = (byte*)normaldata.Scan0;

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

            originalimage.UnlockBits(originaldata);
            normalmap.UnlockBits(normaldata);
        }
        public static Bitmap GenerateNormalMap(Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            Bitmap normalMap = new Bitmap(width, height, PixelFormat.Format24bppRgb);

            // Lock bits for both the input bitmap and the output normal map
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            BitmapData normalMapData = normalMap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            int stride = bitmapData.Stride;
            IntPtr bitmapPtr = bitmapData.Scan0;
            IntPtr normalMapPtr = normalMapData.Scan0;

            unsafe
            {
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
            }

            // Unlock the bits
            bitmap.UnlockBits(bitmapData);
            normalMap.UnlockBits(normalMapData);

            return normalMap;
        }
    }
}