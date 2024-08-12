using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TerrainGenerator
{
    public unsafe class Polygon : IDisposable
    {
        public Point[] points;
        private int* indices;


        [DllImport("polygon.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern byte* drawPolygonWithTriangles(int* x, int* y, int* indices, int numTriangles, byte* color, byte* bitmap, int width, int height);
        public Polygon(Point[] points)
        {
            if (points.Count() <= 2)
            {
                throw new Exception("Polygon must have at least 3 points");
            }
            this.points = points;

            int vertexstart = 0;
            indices = (int*)Marshal.AllocHGlobal(3 * (points.Length - 2) * sizeof(int));

            for (int i = 2; i < points.Length; ++i)
            {
                int idx = (i - 2) * 3;

                //Add the triangle to the list
                indices[idx] = vertexstart;
                indices[idx + 1] = vertexstart + (i - 1);
                indices[idx + 2] = vertexstart + i;
            }
        }
        public void Draw(byte* color, BitmapData bmp)
        {
            int* xs = (int*)Marshal.AllocHGlobal(points.Length * sizeof(int));
            int* ys = (int*)Marshal.AllocHGlobal(points.Length * sizeof(int));

            for (int i = 0; i < points.Length; ++i)
            {
                xs[i] = points[i].X;
                ys[i] = points[i].Y;
            }
            byte* result = drawPolygonWithTriangles(xs, ys, indices, points.Length - 2, color, (byte*)bmp.Scan0, bmp.Width, bmp.Height);
            Buffer.MemoryCopy(result, (byte*)bmp.Scan0, bmp.Stride * bmp.Height, bmp.Stride * bmp.Height);
            Marshal.FreeHGlobal((IntPtr)xs);
            Marshal.FreeHGlobal((IntPtr)ys);
        }
        public void Draw(Color color, BitmapData bmp)
        {
            byte* bytecolor = (byte*)Marshal.AllocHGlobal(4);
            bytecolor[0] = color.B;
            bytecolor[1] = color.G;
            bytecolor[2] = color.R;
            bytecolor[3] = color.A;
            Draw(bytecolor, bmp);
            Marshal.FreeHGlobal((IntPtr)bytecolor);
        }

        public void Dispose()
        {
            Marshal.FreeHGlobal((IntPtr)indices);
        }
    }
}