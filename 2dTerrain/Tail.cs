using System.Runtime.Serialization;

namespace TerrainGenerator
{
    partial class Caterpillar
    {
        public class Tail
        {
            //A swinging mace on the end of the caterpillar
            public int length;
            public int sectionwidth;
            public int sectionheight;
            public PointF start;
            public PointF[] points;
            public PointF[] start_points;

            public Tail(int length, PointF start, int sectionwidth, int sectionheight)
            {
                this.length = length;
                this.start = start;
                this.sectionwidth = sectionwidth;
                this.sectionheight = sectionheight;
                points = new PointF[length];
                start_points = new PointF[length];
                for (int i = 1; i < length; i++)
                {
                    points[i] = new PointF(start.X - i * sectionwidth, start.Y);
                }
            }
            int time = 0;
            public void Swing()
            {
                ++time;
                points[0] = start_points[0];
                for (int i = 1; i < length; ++i)
                {
                    double angle = CalculateAngle(start_points[i - 1], start_points[i]) + Math.PI / 2;
                    double modifier = Math.Sin(time / 30.0);
                    const double swingamt = 1;

                    // Calculate the swingsize based on the curve of a circle
                    double normalized_i = (double)i / length;
                    double curve_value = Math.Sin(normalized_i * Math.PI);  // Curve that peaks at i == length/2
                    curve_value = i*i;
                    double swingsize = modifier * curve_value * swingamt;

                    const double paralellswingcutoff = 0.1;
                    if (modifier > paralellswingcutoff)
                    {
                        //Swing perpindicular 
                        points[i] = new PointF(start_points[i].X + (float)(Math.Cos(angle) * swingsize),
                                               start_points[i].Y + (float)(Math.Sin(angle) * swingsize));
                        //Swing paralell
                        modifier -= paralellswingcutoff;
                        var paralell_swingsize = modifier * curve_value * swingamt;
                        points[i] = new PointF(points[i].X - (float)(Math.Cos(angle - Math.PI / 2) * paralell_swingsize),
                                               points[i].Y - (float)(Math.Sin(angle - Math.PI / 2) * paralell_swingsize));
                    }
                    else if (modifier < -paralellswingcutoff)
                    {
                        //Swing perpindicular
                        points[i] = new PointF(start_points[i].X + (float)(Math.Cos(angle) * swingsize),
                                               start_points[i].Y + (float)(Math.Sin(angle) * swingsize));
                        //Swing paralell
                        modifier += paralellswingcutoff;
                        var paralell_swingsize = modifier * curve_value * swingamt;
                        points[i] = new PointF(points[i].X + (float)(Math.Cos(angle - Math.PI / 2) * paralell_swingsize),
                                               points[i].Y + (float)(Math.Sin(angle - Math.PI / 2) * paralell_swingsize));
                    }
                    else
                    {
                        points[i] = new PointF(start_points[i].X + (float)(Math.Cos(angle) * swingsize),
                                               start_points[i].Y + (float)(Math.Sin(angle) * swingsize));
                    }
                }
            }
        }
    }
}