using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using System.Windows.Forms.VisualStyles;
using TerrainGenerator;

namespace _2dTerrain
{
    public partial class Form1 : Form
    {
        PictureBox pictureBox = new PictureBox();
        Button generateButton = new Button();
        Bitmap result = new Bitmap(1,1);
        public Form1()
        {
            InitializeComponent();

            this.Size = Screen.PrimaryScreen.WorkingArea.Size;

            pictureBox.Location = new Point(0, 0);
            pictureBox.Size = new Size(Width, Height);
            pictureBox.Paint += DrawMap;
            generateButton.Location = new Point(Width - 100, Height - 70); 
            generateButton.Size = new Size(80, 30);
            generateButton.Text = "Generate";
            generateButton.Click += GenerateTiles;
            Controls.Add(generateButton);
            Controls.Add(pictureBox);
        }
        public unsafe void GenerateTiles(object sender, EventArgs e)
        {
            const int wallwidth = 100;
            const int wallheight = 100;
            const int brickwidth = 20;
            const int brickheight = 10;
            const int rockdist = 1;

            result = new Bitmap(Width, Height);
            Graphics g = Graphics.FromImage(result);
            //Create a gridish style pattern of rocks
            Lake[,] lakes = new Lake[wallwidth, wallheight];
            for (int y = 0; y < wallheight; ++y)
            {
                int xoffset = 0;
                for (int x = 0; x < wallwidth; ++ x)
                {
                    Random r = new Random();
                    var lakerect = new Rectangle(0, 0, (int)(brickwidth * (r.NextDouble()/2 + 0.5f)), (int)(brickheight * (r.NextDouble() / 2 + 0.5f))); //Create a rock at 0,0

                    Lake lake = Lake.GenerateLake(lakerect, 20);

                    lakes[x,y] = lake;
                    int yoffset = 0; //Statically offset at 50 to not overflow top
                    if (y >= 1) //Dont look at row above it already on top row
                    {
                        yoffset = lakes[x, y-1].bounds.Max(p=>p.Y) + rockdist; //Find the lowest point on the rock above
                        int furtheresttop = lake.bounds.Min(p => p.Y); //Find the highest point on our rock
                        yoffset -= furtheresttop; //Modify the offset to make sure we dont intersect

                    }
                    int furtherestleft = lake.bounds.Min(p => p.X); //Find the furtherest left point on us
                    xoffset -= furtherestleft; //Modify the offset to make sure we dont intersect
                    for(int i = 0; i < lake.bounds.Count(); ++i)
                    {
                        var point = lake.bounds[i];
                        lake.bounds[i] = new Point(point.X + xoffset, point.Y + yoffset); //Apply offset
                    }

                    //g.FillPolygon(new Pen(Color.DarkGray).Brush, lake.bounds.ToArray()); //Draw rock
                    xoffset = rockdist + lake.bounds.Max(p => p.X); //Make the xoffset for the next rock the furtherest right point on this rock
                    
                    g.DrawRectangle(new Pen(Color.Black), new Rectangle(xoffset - lakerect.Width, yoffset, lakerect.Width, lakerect.Height));
                }
            }

            //Generate rock textures
            //var writebmpdata = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);

            //byte* writeptr = (byte*)(writebmpdata.Scan0);
            /*
            for (int x = 0; x < result.Width; ++x)
            {
                for (int y = 0; y < result.Height; ++y)
                {
                    foreach (var lake in lakes)
                    {
                        if (PointInPolygon(x, y, lake.bounds.ToArray()))
                        {
                            int offset = x * 4 + y * writebmpdata.Stride;
                            Buffer.MemoryCopy(readptr + offset, writeptr + offset, 4, 4);
                        }
                    }
                }
            }*/
            //result.UnlockBits(writebmpdata);

            pictureBox.Invalidate();
            //pictureBox.Image = result;
        }
        public bool PointInPolygon(int x, int y, Point[] polygon)
        {
            int polygonLength = polygon.Length, i = 0;
            bool inside = false;
            // x, y for tested point.
            float pointX = x, pointY = y;
            // start / end point for the current polygon segment.
            float startX, startY, endX, endY;
            PointF endPoint = polygon[polygonLength - 1];
            endX = endPoint.X;
            endY = endPoint.Y;
            while (i < polygonLength)
            {
                startX = endX; startY = endY;
                endPoint = polygon[i++];
                endX = endPoint.X; endY = endPoint.Y;
                //
                inside ^= (endY > pointY ^ startY > pointY) /* ? pointY inside [startY;endY] segment ? */
                          && /* if so, test if it is under the segment */
                          ((pointX - endX) < (pointY - endY) * (startX - endX) / (startY - endY));
            }
            return inside;
        }
        struct ColorArea
        {
            public Color color;
            public double upperbound;

            public ColorArea(Color color, double upperbound)
            {
                this.color = color;
                this.upperbound = upperbound;
            }
        }
        ColorArea[] areas ={new ColorArea(Color.FromArgb(36, 134, 255) , 0.3),
                            new ColorArea(Color.FromArgb(246,215,176)  , 0.4),
                            new ColorArea(Color.FromArgb(77, 122, 77)  , 0.7),
                            new ColorArea(Color.FromArgb(119, 125, 119), 1)};
        public unsafe void GenerateMap(object sender, EventArgs e)
        {
            result = new Bitmap(Width, Height);
            var noise = PerlinNoise.GenerateWhiteNoise(Width,Height);
            var perlin = PerlinNoise.GeneratePerlinNoise(Width, Height, 9);
            var data = result.LockBits(new Rectangle(0,0,Width, Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            
            for (int x = 0; x < Width; ++x)
            {
                for (int y = 0; y < Height; ++y)
                {
                    byte* pixel = ((byte*)data.Scan0) + x * 4 + y * data.Stride;
                    pixel[3] = 255;
                    double height = perlin[x][y];
                    for (int i = 0; i < areas.Length; i++)
                    {
                        ColorArea area = areas[i];
                        if (height < area.upperbound)
                        {
                            Color mycolor = Color.FromArgb(area.color.R, area.color.G, area.color.B);
                            double areaheight = area.upperbound;
                            if (i >= 1) //Blend downwards
                            {
                                ColorArea lowerarea = areas[i-1];
                                areaheight -= lowerarea.upperbound;
                            }

                            pixel[0] = area.color.B;
                            pixel[1] = area.color.G;
                            pixel[2] = area.color.R;
                            break;
                        }
                    }
                }
            }
            result.UnlockBits(data);
            pictureBox.Invalidate();
        }
        public static Color BlendColors(Color color1, Color color2, float blendFactor)
        {
            // Clamp blendFactor between 0 and 1
            blendFactor = Math.Max(0, Math.Min(1, blendFactor));

            // Calculate new color components
            int newRed = (int)(color1.R * (1 - blendFactor) + color2.R * blendFactor);
            int newGreen = (int)(color1.G * (1 - blendFactor) + color2.G * blendFactor);
            int newBlue = (int)(color1.B * (1 - blendFactor) + color2.B * blendFactor);

            // Return the new blended color
            return Color.FromArgb(newRed, newGreen, newBlue);
        }
        public void DrawMap(object sender, PaintEventArgs e)
        {
            e.Graphics.DrawImage(result, 0,0,Width,Height);
        }
    }
}