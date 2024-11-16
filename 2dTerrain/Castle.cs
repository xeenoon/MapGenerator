using System.Drawing.Drawing2D;

namespace TerrainGenerator
{
    public class Castle
    {
        Rectangle bounds;
        public Castle(Rectangle bounds)
        {
            this.bounds = bounds;
        }
        public void Draw(Bitmap b)
        {
            Graphics g = Graphics.FromImage(b);
            g.FillRectangle(new Pen(Color.Blue).Brush, new Rectangle(100, 100, 100, 100));
        }
    }
    public class Tower
    {
        public Rectangle bounds;
    }
}