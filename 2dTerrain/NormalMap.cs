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
        }
    }
}