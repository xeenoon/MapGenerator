using System.Drawing.Drawing2D;

namespace TerrainGenerator
{
    public class Tower
    {
        public Rectangle bounds;
        PointF centre;
        public int brickwidth;
        public float angle;

        public Tower(Rectangle bounds, int brickwidth, float angle)
        {
            this.bounds = bounds;
            centre = new PointF(bounds.X + bounds.Width / 2, bounds.Y + bounds.Height / 2);
            this.brickwidth = brickwidth;
            this.angle = angle;
        }
        public void Draw(Bitmap result)
        {
            int segments = 20;  // The number of curved bricks to draw

            // Calculate the semi-major and semi-minor axes based on the width and height
            float semi_major_axis = bounds.Height / 2 - brickwidth;
            float semi_minor_axis = bounds.Width / 2 - brickwidth;

            // Calculate the angle of rotation for the oval (assuming you want to rotate the oval based on the walkway's angle)
            float rotationOffset = angle;  // Or use another value for rotation

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
                    // Scale the texture to 0.2x resolution
                    var scaleMatrix = new Matrix();
                    textureBrush.Transform = scaleMatrix;

                    // Create a rotation matrix to rotate the texture and ellipse bounds
                    Matrix rotationMatrix = new Matrix();
                    rotationMatrix.RotateAt(rotationOffset * 180f / (float)Math.PI, centre);  // Rotate around the center of the ellipse

                    // Apply the rotation matrix to the canvas and texture brush
                    canvas.Transform = rotationMatrix;

                    rotationMatrix.Scale(0.2f, 0.2f); // Apply 0.1x scaling
                    textureBrush.Transform = rotationMatrix;

                    // Fill the rotated ellipse with the scaled and rotated texture
                    canvas.FillEllipse(textureBrush, ellipseBounds);
                }
            }

            // Calculate the angle increment for each segment
            float angleIncrement = (float)(2 * Math.PI / segments);

            // Draw the curved bricks along the rotated oval
            for (double angle = 0; angle < 2 * Math.PI; angle += angleIncrement)
            {
                // Calculate the start and end positions based on the angle
                PointF start = centre.Add(new PointF((float)Math.Cos(angle) * semi_major_axis, (float)Math.Sin(angle) * semi_minor_axis));
                PointF end = centre.Add(new PointF((float)Math.Cos(angle + angleIncrement) * semi_major_axis, (float)Math.Sin(angle + angleIncrement) * semi_minor_axis));

                // Create and draw the curved brick using the CurvedBrick constructor
                CurvedBrick curvedBrick = new CurvedBrick(
                    (int)semi_major_axis,
                    (int)semi_minor_axis,
                    start,
                    end,
                    centre,
                    brickwidth * 2,
                    rotationOffset,
                    (float)start.DistanceTo(end)
                );
                curvedBrick.Draw(result);  // Draw the curved brick on the result bitmap
            }
        }


    }
}