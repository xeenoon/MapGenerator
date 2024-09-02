using System.ComponentModel.DataAnnotations;

namespace TerrainGenerator
{
    partial class Caterpillar
    {
        public class Mouth
        {
            public PointF head;
            public PointF jawtop;
            public PointF jawbot;
            public PointF start_jawtop;
            public PointF start_jawbot;
            public PointF jawmid;
            public int jawlength;
            public static double jawangle = Math.PI/4;
            public Mouth(PointF head, int jawlength)
            {
                this.jawlength = jawlength;
                jawmid = new PointF(head.X + jawlength, head.Y);
                this.head = head;

                // Vector from head to jawmid
                var hf = new PointF(jawmid.X - head.X, jawmid.Y - head.Y);

                // Length of the vector
                float length = (float)Math.Sqrt(hf.X * hf.X + hf.Y * hf.Y);

                // Calculate angles in radians
                float angle = (float)Math.Atan2(hf.Y, hf.X);

                // 30 degrees in radians (use radians directly)
                float d_angle = (float)(jawangle);

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

                start_jawbot = jawbot;
                start_jawtop = jawtop;
            }
            int time = 0;
            public void Bite(double speed)
            {
                time++;
                double sinvalue = Math.Sin(speed * time / 70.0);
                sinvalue *= sinvalue;

                var jawtopangle = CalculateAngle(head, start_jawtop);
                jawtopangle -= jawangle * Math.Abs(sinvalue);
                jawtop = new PointF((float)(Math.Cos(jawtopangle) * jawlength + head.X), (float)(Math.Sin(jawtopangle) * jawlength + head.Y));
                var jawbotangle = CalculateAngle(head, start_jawbot);
                jawbotangle += jawangle * Math.Abs(sinvalue);

                jawbot = new PointF((float)(Math.Cos(jawbotangle) * jawlength + head.X), (float)(Math.Sin(jawbotangle) * jawlength + head.Y));
                return;
            }
        }
    }
}