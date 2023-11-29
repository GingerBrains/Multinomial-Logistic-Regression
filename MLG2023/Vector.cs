using System;
using System.Collections.Generic;
using System.Text;

namespace MLG2023
{
    public class Vector
    {
        private double[] data = new double[3];
        public int Size //read-only property
        { get { return data.Length; } }

        public Vector()
        {

        }

        public Vector(int size)
        {
            if(size < 1)
            {
                Console.WriteLine("Invalid vector size {0}, set to default of 3 instead", size);
                size = 3;
            }
            data = new double[size];
        }

        public Vector(Vector v)
        {
            data = new double[v.Size];
            for (int i = 0; i < this.Size; i++)
                data[i] = v[i];
        }

        public double this[int i]
        {
            get { return data[i]; }
            set { data[i] = value; }
        }

        public static Vector operator+(Vector lhs, Vector rhs)
        {
            int i = 0;
            Vector res = new Vector(lhs.Size);
            for (i = 0; i < lhs.Size; i++)
                res[i] = lhs[i] + rhs[i];
            return res;
        }

        public static Vector operator +(Vector lhs)
        {
            int i = 0;
            Vector res = new Vector(lhs.Size);
            for (i = 0; i < lhs.Size; i++)
                res[i] = lhs[i];
            return res;
        }

        public static Vector operator -(Vector lhs, Vector rhs)
        {
            int i = 0;
            Vector res = new Vector(lhs.Size);
            for (i = 0; i < lhs.Size; i++)
                res[i] = lhs[i] - rhs[i];
            return res;
        }

        public static Vector operator -(Vector lhs)
        {
            int i = 0;
            Vector res = new Vector(lhs.Size);
            for (i = 0; i < lhs.Size; i++)
                res[i] = -lhs[i];
            return res;
        }

        public static double operator*(Vector lhs, Vector rhs)
        {
            int i;
            double res = 0;
            for (i = 0; i < lhs.Size; i++)
                res += lhs[i]*rhs[i];
            return res;
        }
        public void Zero()
        {
            int i = 0;            
            for (i = 0; i < this.Size; i++)
                this[i] = 0;
        }
        public override string ToString()
        {
            int i;
            string tmp = "(";
            for (i = 0; i < this.Size - 1; i++)
            {
                tmp += string.Format("{0:F2}, ", this[i]);
            }
            tmp += string.Format("{0:F2}", this[i]);
            tmp += ")";
            return tmp;
        }
    }
}
