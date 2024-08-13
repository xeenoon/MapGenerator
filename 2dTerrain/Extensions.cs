public static class Extensions
{
    public static Stack<int> Shuffle(this Stack<int> ints) //Shuffles deck
    {
        Random r = new Random();
        Dictionary<int, int> intindices = new Dictionary<int, int>();
        foreach (var i in ints)
        {
            int rand = 0;
            do
            {
                rand = r.Next();
            } while (intindices.ContainsKey(rand)); //Ensure no duplicate keys despite the fact that the chance is literally 1/4 billion
            intindices.Add(rand, i);
        }
        return new Stack<int>(intindices.OrderBy(c => c.Key).Select(c => c.Value).ToList());
    }
}