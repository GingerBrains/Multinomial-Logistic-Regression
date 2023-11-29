using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace MLG2023
{
    public static class RNG
    {
        private static Random r = new Random();
        public static int GetRandomInt(int n)
        {
            return r.Next(n);
        }

        public static double GetRandomDouble(double a, double b)
        {
            return a + (b - a) * r.NextDouble();
        }
    }
    public class MLRRow //class specifically for MLR
    {
        public MLRRow(DataRow r) 
        {
            int i = 0;
            //Numin -> number of inputs
            //Numout -> number of outputs
            vin = new Vector(r.Numin);
            for (i = 0; i < vin.Size; i++)
                vin[i] = r[IOType.input, i];
            vout = new Vector(r.Numout);
            for (i = 0; i < vout.Size; i++)
                vout[i] = r[IOType.output, i];

        }
        private Vector vin;
        public Vector Vin
        { 
            get { return vin; }
            set { vin = value; }
        }
        private Vector vout;
        public Vector Vout
        { 
            get { return vout; } 
            set { vout = value; } 
        }

    }
    public class MLR
    {
        private List<MLRRow> alldata, traindata, testdata;
        private double split = 0.8;
        //actual weights and biases
        private List<Vector> weights;
        private List<double> biases;
        //for adjustments
        private List<Vector> weights_adj;
        private List<double> biases_adj;
        private int ninputs;
        private int noutputs;
        private int nepochs = 1000;
        public int Nepochs
        { 
            get { return nepochs; }
            set { nepochs = value; }
        }
        //batchsize = 1 means stochastic
        //batchsize = nepochs batch
        //batchsize < nepochs min-batch
        private int batchSize = 1;
        public int BatchSize
        {
            get { return batchSize; }
            set { batchSize = value; }
        }
        private double learningRate = 0.1;
        public double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }
        
        public double Split
        {
            get { return split; }
            set { split = value; }
        }
        public MLR() { }

        public void ImportData(List<DataRow> data)
        {
            MLRRow r = null; 
            int i,j;
            alldata = new List<MLRRow>();
            //import and convert to our List<MLRRow>
            foreach (DataRow row in data)
            {
                r = new MLRRow(row);
                alldata.Add(r);
            }
            ninputs= r.Vin.Size;
            noutputs= r.Vout.Size;
            //set up the weights and biases
            weights = new List<Vector>();
            for (i = 0; i < noutputs; i++)
            {
                var v = new Vector(ninputs);
                for (j = 0; j < ninputs; j++)
                    v[j] = RNG.GetRandomDouble(0, 1);
                weights.Add(v);
            }
            biases = new List<double>();
            for (i = 0; i < noutputs; i++)
            {                                
                var tmp = RNG.GetRandomDouble(0, 1);
                biases.Add(tmp);
            }
            //set up the weights and biases adjustments
            weights_adj = new List<Vector>();
            for (i = 0; i < noutputs; i++)
            {
                var v = new Vector(ninputs);
                weights_adj.Add(v);
            }
            biases_adj = new List<double>();
            for (i = 0; i < noutputs; i++)
            {                
                biases_adj.Add(0);
            }
        }

        public Vector ProcessRow(MLRRow row)
        {
            int count = 0;
            Vector z = new Vector(noutputs);
            //compute estimate
            foreach (Vector w in weights)
            {
                z[count] = row.Vin * w + biases[count];
                count++;
            }
            var yhat = SoftMax(z);
            return yhat;
        }

        public Vector SoftMax(Vector v)
        {
            int i;
            Vector tmp = new Vector(v.Size);
            double sum = 0;
            for(i = 0; i < v.Size;i++)
                sum += Math.Exp(v[i]);
            for (i = 0; i < v.Size; i++)
                tmp[i] = Math.Exp(v[i])/sum;
            return tmp;
        }

        public void Train()
        {
            int i = 0;
            int count = 0;
            Vector yhat;
            for(i = 0; i < nepochs;i++)
            {
                count = 0;
                ShuffleData(traindata);
                foreach(var r in traindata)
                {
                    yhat = ProcessRow(r);
                    ComputeAdjusments(yhat, r);
                    count++;
                    if(count % BatchSize == 0)
                    {
                        UpdateWeightsAndBiases();
                        ResetAdjustments(yhat, r);
                    }
                    else
                    {
                        ComputeAverageAdjustments();
                    }
                }
            }
            //ComputeAverageAdjustmentsAtEnd();
        }

        public void Test()
        {
            List<Vector> yhatValues = new List<Vector>();
            Vector yhat = new Vector(noutputs);
            int yHatMaxIndex,yMaxIndex;
            foreach (var r in testdata)
            {
                yhat = ProcessRow(r);
                yhatValues.Add(yhat);
            }

            for(int i=0 ; i<yhatValues.Count ; i++)
            {
                //yHatMaxIndex = yhatValues[i].ToList().IndexOf(yhatValues[i].Max());
                //yMaxIndex = testdata[i].Vout.ToList().IndexOf(yhatValues[i].Max());
                Console.WriteLine("yhat = {0} \t y = {1}", yhatValues[i], testdata[i].Vout);
            }

        }

        public void ResetAdjustments(Vector yhat, MLRRow row)
        {
            //set all adjustments to 0
            foreach (var v in weights_adj)
                v.Zero();
            for (int i = 0; i < biases_adj.Count; i++)
                biases_adj[i] = 0;
        }
        public void ComputeAdjusments(Vector yhat, MLRRow r)
        {
            int i = 0;
            int count = 0;

            //calc adjustments
            //each v is of length ninputs
            //count is the class
            foreach (var v in weights_adj)
            {
                for (i = 0; i < v.Size; i++)
                    v[i] = (r.Vout[count] - yhat[count]) * r.Vin[i];
                count++;
            }
            for (i = 0; i < biases_adj.Count; i++)
                biases_adj[i] = (r.Vout[i] - yhat[i]);
        }

        public void UpdateWeightsAndBiases()
        {
            // Update weights using adjustments
            for(int i = 0; i < weights.Count; i++)
            {
                for (int j = 0; j < weights[i].Size; j++)
                {
                    weights[i][j] += LearningRate * weights_adj[i][j];
                }
                biases[i] += LearningRate * biases_adj[i];
            }
        }

        public void ComputeAverageAdjustments()
        {
            // Accumulate adjustments
            for (int i = 0; i < weights_adj.Count; i++)
            {
                for (int j = 0; j < weights_adj[i].Size; j++)
                {
                    weights_adj[i][j] += weights_adj[i][j] / BatchSize;
                }
                biases_adj[i] += biases_adj[i] / BatchSize;
            }
        }


        public void ComputeAverageAdjustmentsAtEnd()
        {
            for (int i = 0; i < weights_adj.Count; i++)
            {
                for (int j = 0; j < weights_adj[i].Size; j++)
                {
                    weights_adj[i][j] /= nepochs * BatchSize;
                }
                biases_adj[i] /= nepochs * BatchSize;
            }
        }



        public void SplitData()
        {
            traindata = new List<MLRRow>();
            testdata = new List<MLRRow>();

            int ntrain =(int)(split*alldata.Count);
            ShuffleData(alldata);
            int i = 0;            
            foreach (var r in alldata)
            {
                if (i < ntrain)
                    traindata.Add(r);
                else
                    testdata.Add(r);
                i++;
            }

        }
        private void ShuffleData(List<MLRRow> data)
        {
            int n = data.Count;
            while (n > 1)
            {
                n--;
                int k = RNG.GetRandomInt(n + 1);
                (data[k], data[n]) = (data[n], data[k]);
            }
        }

    }
}
