namespace TerrainGenerator
{
    public class Mouth
    {
        public PointF head;
        public PointF jawtop;
        public PointF jawbot;
        public double length;
        public Mouth(PointF head, PointF jawmid)
        {
            this.length = jawmid.X - head.X; //TODO refactor
            this.head = head;

            // Vector from head to jawmid
            var hf = new PointF(jawmid.X - head.X, jawmid.Y - head.Y);

            // Length of the vector
            float length = (float)Math.Sqrt(hf.X * hf.X + hf.Y * hf.Y);

            // Calculate angles in radians
            float angle = (float)Math.Atan2(hf.Y, hf.X);

            // 30 degrees in radians (use radians directly)
            float d_angle = (float)(Math.PI / 6.0);

            // Calculate jawtop (-30 degrees from jawmid)
            this.jawtop = new PointF(
                jawmid.X + length * (float)Math.Cos(angle - d_angle),
                jawmid.Y + length * (float)Math.Sin(angle - d_angle)
            );

            // Calculate jawbot (+30 degrees from jawmid)
            this.jawbot = new PointF(
                jawmid.X + length * (float)Math.Cos(angle + d_angle),
                jawmid.Y + length * (float)Math.Sin(angle + d_angle)
            );
        }

    }
}