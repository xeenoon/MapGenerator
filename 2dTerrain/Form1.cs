using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using System.Windows.Forms.VisualStyles;
using TerrainGenerator;
using System.Diagnostics;
using System.Collections.Concurrent;
using System.Timers;

namespace _2dTerrain
{
    public partial class Form1 : Form
    {
        PictureBox pictureBox = new PictureBox();
        Button generateButton = new Button();
        Bitmap result = new Bitmap(1, 1);
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
            generateButton.Click += GenerateCastle;
            Controls.Add(generateButton);
            Controls.Add(pictureBox);
            Rock.Setup();
            //caterpillar = new Caterpillar(50, new Point(600, 500));
            //updatetimer.Elapsed += UpdateTick;
            //updatetimer.AutoReset = false;
            //updatetimer.Start();
        }
        Caterpillar caterpillar;
        System.Timers.Timer updatetimer = new System.Timers.Timer(25);
        bool updateruning = false;
        int ticks = 0;
        public void UpdateTick(object sender, ElapsedEventArgs e)
        {
            ++ticks;
            if (updateruning) { return; }
            updateruning = true;
            result = new Bitmap(Width, Height);
            Point cursorPosition = Cursor.Position;

            // Convert the screen coordinates to client coordinates
            Point clientPosition = this.PointToClient(cursorPosition);

            caterpillar.MoveTowards(clientPosition);
            if (ticks % 10 == 0)
            {
                caterpillar.Grow(1);
            }
            caterpillar.Draw(result);
            pictureBox.Invalidate();
            updateruning = false;
            updatetimer.Start();
        }
        public unsafe void GenerateMapGrid(object sender, EventArgs e)
        {
            result = new Bitmap(Width, Height);
            Graphics g = Graphics.FromImage(result);

            Map map = new Map(200, 100);
            map.GenerateMap();
            for (int x = 0; x < map.width; ++x)
            {
                for (int y = 0; y < map.height; ++y)
                {
                    const int squaresize = 10;
                    Color drawcolor = Color.AliceBlue;
                    var squaretype = (GridSquareType)map.GetGridSquare(x, y);
                    if (squaretype == GridSquareType.Floor)
                    {
                        drawcolor = Color.White;
                    }
                    else if (squaretype == GridSquareType.Wall)
                    {
                        drawcolor = Color.Black;
                    }
                    else if (squaretype == GridSquareType.Marked)
                    {
                        drawcolor = Color.Red;
                    }
                    g.FillRectangle(new Pen(drawcolor).Brush, new Rectangle(x * squaresize, y * squaresize, squaresize, squaresize));
                }
            }
            pictureBox.Invalidate();
        }
        [DllImport("vectorexample.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ExtVectorAdd(IntPtr a, IntPtr b, int n);

        [DllImport("vectorexample.dll", CallingConvention = CallingConvention.Cdecl)]
        public unsafe static extern void FreeMemory(IntPtr ptr);
        public unsafe void TestDLL(object sender, EventArgs e)
        {
            Rock rock = new Rock();
            rock.GenerateShape(new Rectangle(200, 200, 400, 400), 20);
            Point[] shape = rock.bounds.ToArray();
            Polygon polygon = new Polygon(shape);
            result = new Bitmap(Width, Height);
            var bmpdata = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
            polygon.Draw(Color.FromArgb(255, 255, 0, 0), bmpdata);

            result.UnlockBits(bmpdata);
            Graphics g = Graphics.FromImage(result);
            polygon.Move(new Point(600, 0));
            polygon.DebugDraw(g);
            g.DrawPolygon(new Pen(Color.Black, 5), polygon.points);
            g.DrawPolygon(new Pen(Color.Black, 5), shape);

            pictureBox.Invalidate();
        }
        public void GenerateCastle(object sender, EventArgs e)
        {
            result = new Bitmap(Width, Height);
            Castle castle = new Castle(new Rectangle(0,0,Width, Height));
            castle.Draw(result);
            pictureBox.Invalidate();
        }
        public unsafe void GeneratePuddle(object sender, EventArgs e)
        {
            result = new Bitmap(Width, Height);
            Graphics g = Graphics.FromImage(result);

            const int wallwidth = 100;
            const int wallheight = 100;
            const int brickwidth = 25;
            const int brickheight = 25;
            const int rockdist = 2;

            string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            var grout = (Bitmap)Image.FromFile(exePath + "\\images\\dirtseam.jpg");
            int tilesize = 1;
            for (int x = 0; x < Math.Ceiling((double)result.Width / grout.Width) * tilesize; ++x)
            {
                for (int y = 0; y < Math.Ceiling((double)result.Height / grout.Height) * tilesize; ++y)
                {
                    g.DrawImage(grout, new Point(x * grout.Width / tilesize, y * grout.Height / tilesize));
                }
            }


            Rock[,] rocks = new Rock[wallwidth, wallheight];
            var puddle = new Puddle();
            puddle.GenerateShape(new Rectangle(400, 400, 400, 400), 360);

            for (int y = 0; y < wallheight; ++y)
            {
                int xoffset = y % 2 == 0 ? 0 : brickwidth / 2;
                for (int x = 0; x < wallwidth; ++x)
                {
                    Random r = new Random();
                    double scalingdifference = 1.8;
                    var rockrect = new Rectangle(0, 0, (int)(brickwidth * (r.NextDouble() / scalingdifference + (1 - 1 / scalingdifference))),
                    (int)(brickheight * (r.NextDouble() / scalingdifference + (1 - 1 / scalingdifference)))); //Create a rock at 0,0

                    Rock rock = new Rock();
                    rock.GenerateShape(rockrect, 20);

                    rocks[x, y] = rock;

                    int furtherestleft = rock.bounds.Min(p => p.X); //Find the furtherest left point on us
                    xoffset -= furtherestleft; //Modify the offset to make sure we dont intersect
                    int furtherestright = rock.bounds.Max(p => p.X);
                    for (int i = 0; i < rock.bounds.Count(); ++i)
                    {
                        var point = rock.bounds[i];
                        rock.bounds[i] = new Point(point.X + xoffset, point.Y); //Apply offset
                    }
                    xoffset += furtherestright; //Make the xoffset for the next rock the furtherest right point on this rock
                                                //Make sure to update after placing the rock

                    int yoffset = 0;
                    Point lowest_point_above_rock = new Point(0, 0);
                    Point top_point_m_rock = new Point(0, 0);
                    if (y >= 1) //Dont look at row above it already on top row
                    {
                        //Cast rays up from every point until we intersect
                        List<Rock> rocksAbove = new List<Rock>();

                        // Check all rocks above the current rock
                        for (int i = 0; i < wallwidth; i++)
                        {
                            int j = y - 1;
                            Rock aboveRock = rocks[i, j];

                            if (aboveRock != null)
                            {
                                // Check each point in the current rock's bounds
                                foreach (var point in rock.bounds)
                                {
                                    // Cast a ray upwards from this point
                                    for (int i1 = 0; i1 < aboveRock.bounds.Count; i1++)
                                    {
                                        Point p0 = aboveRock.bounds[i1];
                                        Point p1 = aboveRock.bounds[i1 == aboveRock.bounds.Count - 1 ? 0 : i1 + 1];

                                        if ((p0.X < point.X && p1.X > point.X) ||
                                            (p1.X < point.X && p0.X > point.X))
                                        {
                                            if (!rocksAbove.Contains(aboveRock))
                                            {
                                                rocksAbove.Add(aboveRock);
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        Point lowestaboverocks = rocksAbove.SelectMany(r => r.bounds).ToList().OrderByDescending(p => p.Y).FirstOrDefault();
                        Point highestmyrock = rock.bounds.OrderBy(p => p.Y).FirstOrDefault();
                        int starty = lowestaboverocks.Y - highestmyrock.Y; //Find the points that the rocks would touch if the orientation was worst case possible
                        List<int> distances = new List<int>();
                        //Cast rays from every single point upwards and find how far away they are from the above rocks
                        for (int pointidx = 0; pointidx < rock.bounds.Count(); ++pointidx)
                        {
                            var point = rock.bounds[pointidx];
                            point = new Point(point.X, point.Y + starty);

                            //Check if casting upwards would intersect with the current rock, i.e. in the bottom half
                            if (pointidx <= rock.bounds.Count() / 2)
                            {
                                continue;
                            }

                            // Cast a ray upwards from this point
                            foreach (var aboveRock in rocksAbove)
                            {
                                for (int i1 = 0; i1 < aboveRock.bounds.Count; i1++)
                                {
                                    //Remove the top half points for optimization
                                    if (i1 >= aboveRock.bounds.Count() / 2)
                                    {
                                        continue;
                                    }
                                    Point p0 = aboveRock.bounds[i1];
                                    Point p1 = aboveRock.bounds[i1 == aboveRock.bounds.Count - 1 ? 0 : i1 + 1];

                                    if (p0.X < point.X && p1.X > point.X ||
                                        p1.X < point.X && p0.X > point.X)
                                    {
                                        var distance = (int)DistanceToLine(new Point(point.X, point.Y), new Line(p0, p1)).Y;

                                        //Visual debug

                                        //g.FillEllipse(new Pen(Color.Red).Brush, new Rectangle(point.X - 5, point.Y - 5, 10, 10));
                                        //g.DrawLine(new Pen(Color.Red), point, p0);
                                        //g.DrawLine(new Pen(Color.Red), point, p1);
                                        //g.FillEllipse(new Pen(Color.Red).Brush, new Rectangle(lowest_point_above_rock.X - 5, lowest_point_above_rock.Y - 5, 10, 10));
                                        //g.FillEllipse(new Pen(Color.Red).Brush, new Rectangle(top_point_m_rock.X - 5 + xoffset, top_point_m_rock.Y - 5 + yoffset, 10, 10));

                                        distances.Add(distance);
                                    }
                                }
                            }
                        }
                        int lowestdistance = distances.OrderBy(d => d).FirstOrDefault();
                        //MessageBox.Show(lowestdistance.ToString());
                        starty -= lowestdistance;
                        starty += 10;
                        yoffset = starty;
                    }

                    for (int i = 0; i < rock.bounds.Count(); ++i)
                    {
                        var point = rock.bounds[i];
                        rock.bounds[i] = new Point(point.X, point.Y + yoffset); //Apply offset
                    }
                    if (rock.bounds.ToArray().Intersects(puddle.bounds.ToArray()))
                    {
                        continue;
                    }
                    //rock.Draw(result);
                    //g.FillPolygon(new Pen(Color.DarkGray).Brush, rock.bounds.ToArray()); //Draw rock
                }
            }


            //Draw the moss overlay
            var mossbitmapdata = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            Moss moss = new Moss(mossbitmapdata);
            moss.OverlayMoss(0.35, 7);
            result.UnlockBits(mossbitmapdata);

            puddle.DrawPuddle(result);
            //puddle.bakeddistances.UnlockBits(puddle.bakeddistances_data);
            //g.DrawImage(puddle.bakeddistances, new Point(puddle.bakedrectangle.Left, puddle.bakedrectangle.Top));

            //g.DrawPolygon(new Pen(Color.Red), puddle.bounds.ToArray());

            NormalMap normalMap = new NormalMap(NormalMap.GenerateNormalMap(result, 0.5f, puddle), result);
            normalMap.ApplyNormalMap();

            //g.FillPolygon(new Pen(Color.Black).Brush, puddle.bounds.ToArray());
            pictureBox.Invalidate();
        }
        public unsafe void GenerateTiles(object sender, EventArgs e)
        {
            Stopwatch s = new Stopwatch();
            s.Start();
            Rock.Setup();
            const int wallwidth = 50;
            const int wallheight = 30;
            const int brickwidth = 50;
            const int brickheight = 50;
            const int rockdist = 3;

            result = new Bitmap(Width, Height);
            Graphics g = Graphics.FromImage(result);


            string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            var grout = (Bitmap)Image.FromFile(exePath + "\\images\\dirtseam.jpg");
            int tilesize = 1;
            for (int x = 0; x < Math.Ceiling((double)result.Width / grout.Width) * tilesize; ++x)
            {
                for (int y = 0; y < Math.Ceiling((double)result.Height / grout.Height) * tilesize; ++y)
                {
                    g.DrawImage(grout, new Point(x * grout.Width / tilesize, y * grout.Height / tilesize));
                }
            }

            s.Stop();
            long setuptime = s.ElapsedMilliseconds;
            s.Restart();
            //Create a gridish style pattern of rocks
            Rock[,] rocks = new Rock[wallwidth, wallheight];

            ConcurrentBag<Rock> paralellrocks = new ConcurrentBag<Rock>();
            const int rocksperthread = 5;
            Parallel.For(0, wallwidth * wallheight, i =>
            {
                Random r = new Random();
                double scalingdifference = 1.8;

                var rockrect = new Rectangle(0, 0, (int)(brickwidth * (r.NextDouble() / scalingdifference + (1 - 1 / scalingdifference))),
                (int)(brickheight * (r.NextDouble() / scalingdifference + (1 - 1 / scalingdifference)))); //Create a rock at 0,0

                Rock rock = new Rock();
                rock.GenerateShape(rockrect, 20);
                paralellrocks.Add(rock);
            });

            List<Rock> resultrocks = paralellrocks.ToList();
            s.Stop();
            long rockgeneration = s.ElapsedMilliseconds;
            s.Restart();

            for (int y = 0; y < wallheight; ++y)
            {
                int xoffset = y % 2 == 0 ? 0 : brickwidth / 2;
                for (int x = 0; x < wallwidth; ++x)
                {
                    var rock = resultrocks[x + y * wallwidth];
                    rocks[x, y] = rock;

                    int furtherestleft = rock.bounds.Min(p => p.X); //Find the furtherest left point on us
                    xoffset -= furtherestleft; //Modify the offset to make sure we dont intersect
                    xoffset += rockdist;
                    int furtherestright = rock.bounds.Max(p => p.X);
                    for (int i = 0; i < rock.bounds.Count(); ++i)
                    {
                        var point = rock.bounds[i];
                        rock.bounds[i] = new Point(point.X + xoffset, point.Y); //Apply offset
                    }
                    rock.bakedrectangle.X += xoffset;
                    rock.rect_bounds.X += xoffset;
                    rock.centre.X += xoffset;

                    xoffset += furtherestright; //Make the xoffset for the next rock the furtherest right point on this rock
                                                //Make sure to update after placing the rock

                    int yoffset = 0;
                    Point lowest_point_above_rock = new Point(0, 0);
                    Point top_point_m_rock = new Point(0, 0);
                    if (y >= 1) //Dont look at row above it already on top row
                    {
                        //Cast rays up from every point until we intersect
                        List<Rock> rocksAbove = new List<Rock>();

                        // Check all rocks above the current rock
                        for (int i = 0; i < wallwidth; i++)
                        {
                            if (rocksAbove.Count() >= 2)
                            {
                                break; //Assume there wont be more than 2 rocks above
                            }
                            int j = y - 1;
                            int xmod;
                            if (i % 2 == 0)
                            {
                                xmod = i / 2; //Look for fowards rocks
                            }
                            else
                            {
                                xmod = -i / 2; //Look for backwards rocks
                            }
                            if (x + xmod < 0 || x + xmod >= wallwidth)
                            {
                                continue;

                            }
                            Rock aboveRock = rocks[x + xmod, j];

                            if (aboveRock != null)
                            {
                                // Check each point in the current rock's bounds
                                foreach (var point in rock.bounds)
                                {
                                    // Cast a ray upwards from this point
                                    for (int i1 = 0; i1 < aboveRock.bounds.Count; i1++)
                                    {
                                        Point p0 = aboveRock.bounds[i1];
                                        Point p1 = aboveRock.bounds[i1 == aboveRock.bounds.Count - 1 ? 0 : i1 + 1];

                                        if ((p0.X < point.X && p1.X > point.X) ||
                                            (p1.X < point.X && p0.X > point.X))
                                        {
                                            if (!rocksAbove.Contains(aboveRock))
                                            {
                                                rocksAbove.Add(aboveRock);
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        Point lowestaboverocks = rocksAbove.SelectMany(r => r.bounds).ToList().OrderByDescending(p => p.Y).FirstOrDefault();
                        Point highestmyrock = rock.bounds.OrderBy(p => p.Y).FirstOrDefault();
                        int starty = lowestaboverocks.Y - highestmyrock.Y; //Find the points that the rocks would touch if the orientation was worst case possible
                        List<int> distances = new List<int>();
                        //Cast rays from every single point upwards and find how far away they are from the above rocks
                        for (int pointidx = 0; pointidx < rock.bounds.Count(); ++pointidx)
                        {
                            var point = rock.bounds[pointidx];
                            point = new Point(point.X, point.Y + starty);

                            //Check if casting upwards would intersect with the current rock, i.e. in the bottom half
                            if (pointidx <= rock.bounds.Count() / 2)
                            {
                                continue;
                            }

                            // Cast a ray upwards from this point
                            foreach (var aboveRock in rocksAbove)
                            {
                                for (int i1 = 0; i1 < aboveRock.bounds.Count; i1++)
                                {
                                    //Remove the top half points for optimization
                                    if (i1 >= aboveRock.bounds.Count() / 2)
                                    {
                                        continue;
                                    }
                                    Point p0 = aboveRock.bounds[i1];
                                    Point p1 = aboveRock.bounds[i1 == aboveRock.bounds.Count - 1 ? 0 : i1 + 1];

                                    if (p0.X < point.X && p1.X > point.X ||
                                        p1.X < point.X && p0.X > point.X)
                                    {
                                        var distance = (int)DistanceToLine(new Point(point.X, point.Y), new Line(p0, p1)).Y;

                                        //Visual debug

                                        //g.FillEllipse(new Pen(Color.Red).Brush, new Rectangle(point.X - 5, point.Y - 5, 10, 10));
                                        //g.DrawLine(new Pen(Color.Red), point, p0);
                                        //g.DrawLine(new Pen(Color.Red), point, p1);
                                        //g.FillEllipse(new Pen(Color.Red).Brush, new Rectangle(lowest_point_above_rock.X - 5, lowest_point_above_rock.Y - 5, 10, 10));
                                        //g.FillEllipse(new Pen(Color.Red).Brush, new Rectangle(top_point_m_rock.X - 5 + xoffset, top_point_m_rock.Y - 5 + yoffset, 10, 10));

                                        distances.Add(distance);
                                    }
                                }
                            }
                        }
                        int lowestdistance = distances.OrderBy(d => d).FirstOrDefault();
                        //MessageBox.Show(lowestdistance.ToString());
                        starty -= lowestdistance;
                        starty += rockdist;
                        yoffset = starty;
                    }

                    for (int i = 0; i < rock.bounds.Count(); ++i)
                    {
                        var point = rock.bounds[i];
                        rock.bounds[i] = new Point(point.X, point.Y + yoffset); //Apply offset
                    }
                    rock.bakedrectangle.Y += yoffset;
                    rock.rect_bounds.Y += yoffset;
                    rock.centre.Y += yoffset;
                }
            }
            s.Stop();
            long rockmove = s.ElapsedMilliseconds;
            s.Restart();

            var resultbmp = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            Rock.CudaDraw(rocks, wallwidth, wallheight, resultbmp);
            result.UnlockBits(resultbmp);
            s.Stop();
            long rockdraw = s.ElapsedMilliseconds;
            s.Restart();

            //Create the normal map
            NormalMap normalMap = new NormalMap(NormalMap.GenerateNormalMap(result, 1f), result);
            normalMap.ApplyNormalMap();
            s.Stop();
            long normalmilis = s.ElapsedMilliseconds;
            s.Restart();
            //Draw the moss overlay
            var mossbitmapdata = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            Moss moss = new Moss(mossbitmapdata);
            moss.OverlayMoss(0.35, 7);
            result.UnlockBits(mossbitmapdata);

            s.Stop();
            long mossmilis = s.ElapsedMilliseconds;

            pictureBox.Invalidate();
            long totaltime = setuptime + rockgeneration + rockmove + rockdraw + normalmilis + mossmilis;
            string times = $"Setup: {setuptime}\nRock gen: {rockgeneration}\nRock movement: {rockmove}\nRock drawing: {rockdraw}\nNormal: {normalmilis}\nMoss: {mossmilis}\nTotal: {totaltime}";
            MessageBox.Show(times);
        }
        public struct Line
        {
            public PointF start;
            public PointF end;

            public Line(PointF start, PointF end)
            {
                this.start = start;
                this.end = end;
            }
        }
        public static PointF DistanceToLine(PointF p, Line line)
        {
            var A = p.X - line.start.X;
            var B = p.Y - line.start.Y;
            var C = line.end.X - line.start.X;
            var D = line.end.Y - line.start.Y;

            var dot = A * C + B * D;
            var len_sq = C * C + D * D;
            float param = -1;
            if (len_sq != 0) //in case of 0 length line
                param = dot / len_sq;

            float xx, yy;

            if (param < 0)
            {
                xx = line.start.X;
                yy = line.start.Y;
            }
            else if (param > 1)
            {
                xx = line.end.X;
                yy = line.end.Y;
            }
            else
            {
                xx = line.start.X + param * C;
                yy = line.start.Y + param * D;
            }

            var dx = p.X - xx;
            var dy = p.Y - yy;
            return new PointF(dx, dy);
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
            var noise = PerlinNoise.GenerateWhiteNoise(Width, Height);
            var perlin = PerlinNoise.GeneratePerlinNoise(Width, Height, 9);
            var data = result.LockBits(new Rectangle(0, 0, Width, Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

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
                                ColorArea lowerarea = areas[i - 1];
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
            e.Graphics.DrawImage(result, 0, 0, Width, Height);
        }
    }
}