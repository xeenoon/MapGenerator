public class Map
{
    public GridSquare[,] gridSquares;
    public int width;
    public int height;
    public Map(int width, int height)
    {
        this.width = width;
        this.height = height;
        gridSquares = new GridSquare[width, height];
    }
    public void GenerateMap()
    {
        Random r = new Random();
        //Generate a group of rooms
        int maxrooms = 4;
        int builtrooms = 0;
        //Rooms can be from 4x4 to 10x10
        int xlocation;
        int ylocation;
        List<Room> rooms = new List<Room>();
        while (builtrooms < maxrooms)
        {
            int roomwidth = r.Next(4, 11);
            int roomheight = r.Next(4, 11);
            xlocation = r.Next(0, width - roomwidth);
            ylocation = r.Next(0, height - roomheight);
            //Check if we intersect with any other rooms
            Room room = new Room(new Rectangle(xlocation, ylocation, roomwidth, roomheight));
            if (RectangleIntersects(room.bounds, rooms.Select(r => r.bounds).ToList()))
            {
                continue; //Dont have overlapping rooms
            }
            rooms.Add(room);
            builtrooms++;
            //Clear out the whole room location
            for (int x = xlocation; x < xlocation + roomwidth; ++x)
            {
                for (int y = ylocation; y < ylocation + roomheight; ++y)
                {
                    gridSquares[x, y].gridSquareType = GridSquare.GridSquareType.Floor;
                }
            }
        }
        //Connect rooms via hallways
        //Assign doorways to each of the rooms, disallow any other entrances
        foreach (var room in rooms)
        {
            Point random_top_entrance = new Point(room.bounds.X + r.Next(0, room.bounds.Width), room.bounds.Y);
            Point random_bottom_entrance = new Point(room.bounds.X + r.Next(0, room.bounds.Width), room.bounds.Y + room.bounds.Height);
            Point random_left_entrance = new Point(room.bounds.X, room.bounds.Y + r.Next(0, room.bounds.Height));
            Point random_right_entrance = new Point(room.bounds.X + room.bounds.Width, room.bounds.Y + r.Next(0, room.bounds.Height));
            room.doors.Add(random_top_entrance);
            room.doors.Add(random_bottom_entrance);
            room.doors.Add(random_left_entrance);
            room.doors.Add(random_right_entrance);

            for (int x = room.bounds.Left; x < room.bounds.Right; ++x)
            {
                for (int y = room.bounds.Top; y < room.bounds.Bottom; ++y)
                {
                    gridSquares[x, y] = new GridSquare(GridSquare.GridSquareType.Floor);
                }
            }
        }
    }
    public static bool RectangleIntersects(Rectangle rectangle, List<Rectangle> rectangles)
    {
        foreach (var r in rectangles)
        {
            //Check for all corners in my rectangle and see if they are inside the other rectangle
            if (PointInRectangle(new Point(rectangle.X, rectangle.Y), r)) //Top left corner
            {
                return true;
            }
            if (PointInRectangle(new Point(rectangle.X + rectangle.Width, rectangle.Y), r)) //Top left corner
            {
                return true;
            }
            if (PointInRectangle(new Point(rectangle.X, rectangle.Y + rectangle.Height), r)) //Top left corner
            {
                return true;
            }
            if (PointInRectangle(new Point(rectangle.X + rectangle.Width, rectangle.Y + rectangle.Height), r)) //Top left corner
            {
                return true;
            }
        }
        //Nothing intersected?
        return false;
    }
    public static bool PointInRectangle(Point p, Rectangle r) //Also checks if point is on edge
    {
        if (p.X >= r.X && p.X <= r.X + r.Width && p.Y >= r.Y && p.Y <= r.Y + r.Height)
        {
            return true;
        }
        return false;
    }
}
public class Room
{
    public Rectangle bounds;
    public List<Point> doors = new List<Point>();
    public Room(Rectangle bounds)
    {
        this.bounds = bounds;
    }
}
public struct GridSquare
{
    public enum GridSquareType
    {
        Wall,
        Floor,
    }
    public GridSquareType gridSquareType;
    public GridSquare(GridSquareType gridSquareType)
    {
        this.gridSquareType = gridSquareType;
    }
}