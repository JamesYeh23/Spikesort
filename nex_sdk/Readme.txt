# To read .nex files, use the following code:

import NexFileReaders
reader = NexFileReaders.NexFileReader()
fileData = reader.ReadNexFile(r"C:\Data\File1.nex")

# fileData is an object containing data from .nex or .nex5 file. 
# see NexFileData.py for details.

# To read .nex5 files, use the following code:

import NexFileReaders
readerNex5 = NexFileReaders.Nex5FileReader()
fileData = readerNex5.ReadNex5File(r"C:\Data\File1.nex5")

# fileData is an object containing data from .nex or .nex5 file. 
# see NexFileData.py for details.

# To write .nex of .nex5 file, 
# - create a FileData object
# - add data to this object.
# - create NexFileWriter or Nex5FileWriter
# - write data to file

from NexFileData import *
import NexFileWriters

fd = FileData()
fd.Comment = "written by Python NexFileWriter"
fd.TimestampFrequency = 100000
fd.Neurons.append(Neuron('neuron1', [12, 13, 14]))
fd.Events.append(Event('event1', [50, 52]))
fd.Intervals.append(Interval('interval1', [40, 52], [43, 54]))
fd.Markers.append(Marker('marker1', [10, 12], ['f1', 'f2'], [['7', '8'], ['c', 'd']]))
fd.Continuous.append(Continuous('cont1', 1000, [2.1], [0], [127, 129]))
fd.Continuous.append(Continuous('cont2', 10000, [5.1, 42], [0, 3], [127, 129, 22, 23]))
fd.Waveforms.append(Waveform('waveform1', 10000, [5.1, 42], 3, [127, 129, 22, 23, 99, 200]))

writer = NexFileWriters.NexFileWriter()
writer.WriteDataToNexFile(fd, r"C:\Data\writtenByPythonNexFileWriter.nex")

writerNex5 = NexFileWriters.Nex5FileWriter()
writerNex5.WriteDataToNex5File(fd, r"C:\Data\writtenByPythonNexFileWriter.nex5")
