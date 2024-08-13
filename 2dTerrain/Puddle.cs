using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Security.Principal;

namespace TerrainGenerator
{
    public class Puddle : ProceduralShape
    {

        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe void* memcpy(void* dest, void* src, UIntPtr count);

        public unsafe void DrawPuddle(Bitmap result)
        {

            Bitmap polygonmarker = new Bitmap(result.Width, result.Height);
            Graphics graphics = Graphics.FromImage(polygonmarker);
            graphics.FillPolygon(new Pen(Color.FromArgb(255, 255, 0, 0)).Brush, bounds.ToArray()); //Mark the pixel

            var markdata = polygonmarker.LockBits(new Rectangle(0, 0, polygonmarker.Width, polygonmarker.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);
            var markptr = (byte*)markdata.Scan0;

            BitmapData write = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppPArgb);

            string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var mudbmp = (Bitmap)Image.FromFile(exePath + "\\images\\dirtseam.jpg");
            var muddata = mudbmp.LockBits(new Rectangle(0, 0, mudbmp.Width, mudbmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);

            const int watertilefactor = 8;
            var waterbmp = (Bitmap)Image.FromFile(exePath + "\\images\\waterseam.jpg");
            var waterdata = waterbmp.LockBits(new Rectangle(0, 0, waterbmp.Width, waterbmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);


            byte* resultptr = (byte*)write.Scan0;
            byte* mudptr = (byte*)muddata.Scan0;
            byte* waterptr = (byte*)waterdata.Scan0;

            Point topleft = new Point(bounds.Min(p => p.X), bounds.Min(p => p.Y));
            Point botright = new Point(bounds.Max(p => p.X), bounds.Max(p => p.Y));

            for (int x = topleft.X; x < botright.X; ++x)
            {
                for (int y = topleft.Y; y < botright.Y; ++y)
                {
                    byte* result_loc = resultptr + x * 4 + y * 4 * result.Width;
                    byte* mud_loc = mudptr + (x % mudbmp.Width) * 4 + (y % mudbmp.Height) * 4 * mudbmp.Width;
                    byte* water_loc = waterptr + ((x * watertilefactor) % waterbmp.Width) * 4 + ((y * watertilefactor) % waterbmp.Height) * 4 * waterbmp.Width;

                    const double waterblend = 0.5;
                    if (markptr[x * 4 + y * 4 * result.Width + 2] == 255) // Am I in the polygon?
                    {
                        //var distance = (double)DistanceTo(new Point(x, y));
                        var distance = bounds.ToArray().DistanceFromPointToPolygon(new Point(x, y));
                        if (distance <= blenddst && distance != -1) //Blend on edges
                        {
                            //double blendfactor = distance / (blenddst * blendstrength) + (1 - (1.0 / blendstrength));
                            double blendfactor = Math.Min(distance / blenddst, 1);

                            Extensions.BlendColors(result_loc, mud_loc, blendfactor);
                            Extensions.BlendColors(result_loc, water_loc, blendfactor * waterblend);
                        }
                        else
                        {
                            memcpy(result_loc, mud_loc, 3);
                            Extensions.BlendColors(result_loc, water_loc, waterblend);

                        }

                        // Set alpha to 255
                        result_loc[3] = 255;
                    }
                }
            }
            result.UnlockBits(write);
            mudbmp.UnlockBits(muddata);
            polygonmarker.UnlockBits(markdata);
        }
    }
}