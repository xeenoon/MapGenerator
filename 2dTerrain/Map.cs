public class Map
{
    public GridSquare[,] gridSquares;
    public Map(int width, int height)
    {
        gridSquares = new GridSquare[width, height];
    }
}
public class GridSquare
{
    public enum GridSquareType
    {
        Floor,
        Wall,
    }
    public GridSquareType gridSquareType;
    public GridSquare(GridSquareType gridSquareType)
    {
        this.gridSquareType = gridSquareType;
    }
}