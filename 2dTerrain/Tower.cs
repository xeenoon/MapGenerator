using System.Drawing.Drawing2D;

namespace TerrainGenerator
{
    public class Tower
    {
        public Rectangle bounds;
        PointF centre;
        public int brickwidth;

        public Tower(Rectangle bounds, int brickwidth)
        {
            this.bounds = bounds;
            centre = new PointF(bounds.X + bounds.Width / 2, bounds.Y + bounds.Height / 2);
            this.brickwidth = brickwidth;
        }
        public void Draw(Bitmap result)
        {
            int semi_major_axis = bounds.Width / 2 - brickwidth;
            int semi_minor_axis = bounds.Height / 2 - brickwidth;

            // Starting and ending angles for the first brick
            double start_angle = Math.PI / 6;
            double end_angle = Math.PI / 4;

            // Angle increment between bricks
            double angleIncrement = end_angle - start_angle;

            // Loop to draw bricks around the unit circle
            for (double angle = start_angle; angle <= 2 * Math.PI + start_angle; angle += angleIncrement)
            {
                // Calculate the start and end positions based on the angle
                PointF start = centre.Add(new PointF((float)Math.Cos(angle) * semi_major_axis, (float)Math.Sin(angle) * semi_minor_axis));
                PointF end = centre.Add(new PointF((float)Math.Cos(angle + angleIncrement) * semi_major_axis, (float)Math.Sin(angle + angleIncrement) * semi_minor_axis));

                // Create and draw the curved brick
                CurvedBrick curvedBrick = new CurvedBrick(semi_major_axis, semi_minor_axis, start, end, centre, brickwidth * 2);
                curvedBrick.Draw(result);
            }

            string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var wood = (Bitmap)Image.FromFile(exePath + "\\images\\woodplanks.png");

            using (var canvas = Graphics.FromImage(result))
            {
                canvas.SmoothingMode = SmoothingMode.AntiAlias;

                // Define the rectangle that bounds the ellipse
                RectangleF ellipseBounds = new RectangleF(
                    centre.X - semi_major_axis,
                    centre.Y - semi_minor_axis,
                    semi_major_axis * 2,
                    semi_minor_axis * 2);

                // Create a texture brush with the wood texture
                using (TextureBrush textureBrush = new TextureBrush(wood, WrapMode.Tile))
                {
                    // Scale the texture to 0.1x resolution
                    var scaleMatrix = new Matrix();
                    scaleMatrix.Scale(0.2f, 0.2f); // Apply 0.1x scaling
                    textureBrush.Transform = scaleMatrix;

                    // Fill the ellipse with the scaled texture
                    canvas.FillEllipse(textureBrush, ellipseBounds);
                }

            }
        }
    }
}