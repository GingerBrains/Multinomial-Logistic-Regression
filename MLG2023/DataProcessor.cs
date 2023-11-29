using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.ComponentModel.DataAnnotations;
using System.Linq;

namespace MLG2023
{
    public enum FileType
    {
        LR,
        MLR,
        ANN
    }
    public class DataProcessor
    {
        //reads in the data into DataRows
        //all other processing to be done in the specific algorithm
        private FileType ft;
        public FileType Ft
        {
            get { return ft; }
            set { ft = value; }
        }
        private string filelocation;
        public string Filelocation
        {
            get { return filelocation; }
            set { filelocation = value; }
        }
        private bool header;
        public bool Header
        {
            get { return header; }
            set { header = value; }
        }
        //for use in ANN
        private int numinputs;
        public int Numinputs
        {
            get { return numinputs; }
            set { numinputs = value;  }
        }
        private int numoutputs;
        private int numcols;
        private string[] headers;
        private List<DataRow> alldata;
        public List<DataRow> Alldata
        {
            get { return alldata; }
            set { alldata = value; }
        }

        public void LoadData()
        {
            int i,j;
            bool success;
            double tmp;
            string line;
            string[] lineparts= null;
            List<string> categories = new List<string>();
            alldata = new List<DataRow>();
            //check file exists
            if (!File.Exists(filelocation))
            {
                Console.WriteLine("The file at \"{0}\", does not exist",filelocation);
                return;
            }
            StreamReader sr = new StreamReader(filelocation);
            if (header) //read firstline
            {
                line = sr.ReadLine();
                lineparts = line.Split(",");
                headers = new string[lineparts.Length];
                for (i = 0; i < lineparts.Length; i++)
                    headers[i] = string.Copy(lineparts[i]);
            }
            else //read firstline count columns and go back to start
            {
                line = sr.ReadLine();
                lineparts = line.Split(",");
                headers = new string[lineparts.Length];
                for (i = 0; i < lineparts.Length; i++)
                    headers[i] = string.Format("c{0}",i+1);
                //go back to start
                sr.DiscardBufferedData();
                sr.BaseStream.Seek(0, SeekOrigin.Begin);
            }
            numcols= lineparts.Length;
            //create header labels
            switch(ft)
            {
                case FileType.LR:
                    //here we have numcols - 1 data plus one col
                    numinputs = numcols - 1;
                    numoutputs = 1;
                    break;
                case FileType.MLR:
                    //here we have numcols - 1 data plus a category column to be processed
                    numinputs  = numcols - 1;
                    //process the last column
                    while(!sr.EndOfStream)
                    {
                        line = sr.ReadLine();
                        lineparts = line.Split(",");
                        categories.Add(lineparts[lineparts.Length-1]);
                    }
                    categories = categories.Distinct().ToList();
                    numoutputs = categories.Count;
                    //go back to start
                    sr.DiscardBufferedData();
                    sr.BaseStream.Seek(0, SeekOrigin.Begin);
                    if(header) //move forward one
                        sr.ReadLine();
                    break;
                case FileType.ANN:
                    //here we have numinputs data plus the rest are outputs
                    numoutputs= numcols - numinputs;
                    break;
            }
            //we are now ready to read and store the data
            while (!sr.EndOfStream)
            {
                line = sr.ReadLine();
                lineparts = line.Split(",");
                DataRow r = new DataRow(numinputs, numoutputs);
                for (i = 0; i < numinputs; i++)
                {
                    success = double.TryParse(lineparts[i], out tmp);
                    r[IOType.input, i] = tmp;
                }
                switch (ft)
                {
                    case FileType.LR:
                        
                        //final bit
                        success = double.TryParse(lineparts[i], out tmp);
                        r[IOType.output, 0] = tmp;
                        break;
                    case FileType.MLR:
                        j = categories.IndexOf(lineparts[i]);
                        for (i = 0; i < numoutputs; i++)
                        {
                            if (i == j)
                                r[IOType.output, i] = 1;
                            else
                                r[IOType.output, i] = 0;
                        }
                        break;
                    case FileType.ANN:
                        j = i;
                        for (i = 0; i < numoutputs; i++, j++)
                        {
                            success = double.TryParse(lineparts[j], out tmp);
                            r[IOType.output, i] = tmp;
                        }                        
                        break;
                }
                alldata.Add(r);
            }

            sr.Close();
        }     
    }
}
