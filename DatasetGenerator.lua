local torch = require 'torch'
local midi = require 'MIDI'
mtbv = require "midiToBinaryVector"
require 'lfs'



function firstToUpper(str)
    return str:gsub("^%l", string.upper)
end

--Create a table to associate genre names with numerical values.

classifier = {}
classes = {}




--Gather the midi files from the music directory. The SongGroupContainer is neccessary since we want to split
--our data into training and testing for each genre. This was re eliminate the possibility of having too 
--few of training midis compared to testing or vica-versa

function GatherMidiData(BaseDir) 
    --Look to see if we have already saved the data.
    SongData_file = 'SongData.t7'
    if paths.filep(SongData_file) then
    	loaded = torch.load(SongData_file)
	classes = loaded.classes
	classifier = loaded.classifier
	SongGroupContainer = loaded.SongGroupContainer
	return SongGroupContainer
    end

    local SongGroupContainer = {}
    directoryCounter = 0;
    for directoryName in lfs.dir(BaseDir) 
    do 
        if directoryName ~= ".." and directoryName ~= "." and lfs.attributes(BaseDir.."/"..directoryName,"mode") == "directory"
        then
            directoryCounter = directoryCounter + 1
            directoryPath = BaseDir.."/"..directoryName.."/"
            
            local obj = 
            {
                Genre = directoryName,
                Songs = {}
            }

            --classifier[directoryName] = firstToUpper(directoryName)
	    classes[directoryCounter] = firstToUpper(directoryName)
            classifier[directoryName] = directoryCounter
	    --classes[directoryCounter] = directoryCounter


	    --print(directoryName)
            --print(classifier[directoryName])
            
            fileCounter = 0
            for filename in lfs.dir(BaseDir.."/"..directoryName) 
            do FullFilePath = BaseDir.."/"..directoryName.."/"..filename
                if string.find(filename, ".mid")
                then 
                    
                    data = midiToBinaryVec(FullFilePath) 
                    if data ~= nil then
                        fileCounter = fileCounter + 1 
                        obj.Songs[fileCounter] = data
                       --print("DATA: ")
                        --print(data)
                        --print(data:size())
                    end
                end
            end
            SongGroupContainer[directoryName] = obj
            --SerializeData(directoryPath..outputFileName, obj)
        end
    end

    SaveData = 
    {
	SongGroupContainer = SongGroupContainer,
	classes = classes,
	classifier = classifier
    }
    
    torch.save(SongData_file, SaveData)
    return SongGroupContainer
end




function SplitMidiData(data, ratio)
    local trainData = {Labels={}, Songs={}}
    local testData = {Labels={}, Songs={}}
    trainData.size = function() return #trainData.Songs end
    testData.size = function() return #testData.Songs end    


    TrainingCounter = 0
    TestingCounter = 0
    for genreKey,value in pairs(data) do 
        local numTrain = math.floor(#data[genreKey].Songs * ratio)
        local numTest = #data[genreKey].Songs - numTrain
	--print(numTrain)
	--print(numTest)            


        for i=1,numTrain do
          TrainingCounter = TrainingCounter + 1
          --print(#data[genreKey].Songs)
          --print(i)
          --print(genreKey)
          trainData.Songs[TrainingCounter] = data[genreKey].Songs[i]:transpose(1,2):clone()
          trainData.Songs[i] = (trainData.Songs[i] - trainData.Songs[i]:mean())/(trainData.Songs[i]:std())
	  trainData.Labels[TrainingCounter] = classifier[genreKey]
        end
        
        for i=numTrain+1,numTrain+numTest do
            TestingCounter = TestingCounter + 1
            testData.Songs[TestingCounter] = data[genreKey].Songs[i]:transpose(1,2):clone()
            testData.Songs[TestingCounter] = (testData.Songs[TestingCounter] - testData.Songs[TestingCounter]:mean())/(testData.Songs[TestingCounter]:std())
            testData.Labels[TestingCounter] = classifier[genreKey]
	    
        end
    end    
   
    return trainData, testData, classes
end




function GetTrainAndTestData(BaseDir, Ratio)    
    Data = GatherMidiData(BaseDir)
    return SplitMidiData(Data, Ratio)
end


--EXAMPLE USAGE
--trainData, testData = GetTrainAndTestData("./music", .5)
--print(trainData)
--print (testData)
--print (classes)










