namespace TerrainGenerator
{
    public class Caterpillar
    {
        public List<Point> spine = new List<Point>();
        public Caterpillar(int length, Point headpos)
        {
            for (int i = 0; i < length; ++i)
            {
                spine.Add(new Point(headpos.X - 50 * i, headpos.Y));
            }
        }
        public void MoveTowards(Point p)
        {

        }
        public void Draw(Bitmap b)
        {
            Graphics g = Graphics.FromImage(b);
            foreach (Point p in spine)
            {
                int size = 50;
                g.FillEllipse(new Pen(Color.RebeccaPurple).Brush, new Rectangle(p.X - size / 2, p.Y - size / 2, size, size));
            }
        }
    }
}