using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace MLG2023
{
    public enum IOType
    {
        input,
        output
    }
    public class DataRow
    {        
        private double [] inputs;
        public int Numin
        { get { return inputs.Length; } }


        private double[] outputs;
        public int Numout
        { get { return outputs.Length; } }
        public DataRow(int numinputs, int numoutputs)
        {
            inputs = new double[numinputs];
            outputs = new double[numoutputs];
        }
        public double this[IOType io, int index] 
        { get { if (io == IOType.input)
                    return inputs[index];
                else
                    return outputs[index];}
            set {
                if (io == IOType.input)
                    inputs[index] = value;
                else
                    outputs[index] = value;
            }
        }
    }
}
