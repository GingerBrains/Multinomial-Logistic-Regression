using System;

namespace MLG2023
{
    class Program
    {
        static void Main(string[] args)
        {
            //Console.WriteLine("Hello World!");
            //Vector a = new Vector(4);
            //a[0] = 1;
            //a[1] = -5;
            //a[2] = 3.4;
            //a[3] = 2.141;

            //Vector b = new Vector(a);
            //Vector c = a + b; //converts to Vector.operator+(a,b)
            //c = +b;
            //Console.WriteLine("c = {0}", c);
            //test logistic regression
            DataProcessor dp = new DataProcessor();

            //test MLR            
            dp.Ft = FileType.MLR;
            dp.Filelocation = "C:\\Users\\Laurel\\OneDrive\\Documents\\College\\UCC\\AM6007\\MLG2023\\dataset\\Iris.csv";
            dp.Header = true;
            dp.LoadData();
            MLR m = new MLR();
            m.ImportData(dp.Alldata);
            m.Nepochs = 500;
            m.Split = 0.9;
            m.LearningRate = 0.1;
            m.BatchSize = 10;
            m.SplitData();
            m.Train();
            m.Test();
            Console.ReadLine();
        }
    }
}
