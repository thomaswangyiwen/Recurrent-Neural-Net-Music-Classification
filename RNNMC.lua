local torch = require 'torch'
local nn = require "nn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetGenerator'
require 'lfs'
require 'concatModule/concatmodel'



--Step 1: Gather our training and testing data - trainData and testData contain a table of Songs and Labels
trainData, testData, classes = GetTrainAndTestData("./music", .8)


--Step 2: Create the model
model = GenerateBagOfClassifiers(#classes)
print(model)

--Step 3: Defne Our Loss Function
criterion = nn.MultiMarginCriterion(2)--nn.ClassNLLCriterion()

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
print(confusion)

-- Log results to files
trainLogger = optim.Logger(paths.concat('.', 'train.log'))
testLogger = optim.Logger(paths.concat('.', 'test.log'))


parameters = {}
gradParameters = {}
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
-- Update we now need to capture all gradients with inside the model......
print("TEST_SEARCH")
if model then
   parameters,gradParameters = model:getParameters()
end




optimState = {
    learningRate = 0.003,
    weightDecay = 0.01,
    momentum = .01,
    learningRateDecay = 5e-7
}
optimMethod = optim.sgd

epoch = 1
batch_size = 32
function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trainData:size())

    
   for t = 1, trainData:size(),batch_size do
   
      xlua.progress(t, trainData:size())  
      local inputs = {}
      
      --print(inputs[0]:size(2))
      local targets = {}
      for s=0,batch_size
      do
      if t+s > trainData:size() then
      break
      end
      inputs[s] = trainData.Songs[shuffle[t+s]]
      targets[s] = trainData.Labels[shuffle[t+s]]
      end  
        
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0
                       local spl_counter = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                     
                          spl_counter  = spl_counter+1
                          local output = model:forward(inputs[i])
                          --output = torch.Tensor(output)
                          local err = criterion:forward(output, targets[i])
                          --print(err)
                          f = f + err

                          --print(output)
                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                        --end
                       --end
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(spl_counter)
                       f = f/spl_counter

                       -- return f and df/dX
                       return f,gradParameters
                    end
        optim.sgd(feval, parameters, optimState)
   end
    
   -- time taken
   time = sys.clock() - time
   time = time / #trainData
   --print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if true then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      --trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat('.', 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end


function test()
   -- local vars
   local time = sys.clock()

   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
  print(testData:size())
   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.Songs[t]
      --input = input:double()
      local target = testData.Labels[t]
      --local sum = torch.Tensor(3)
      --preds = {}
      -- test sample
      --local splitted = input:split(spl,1)
      ---                          for j = 1,#splitted do
         --                 if splitted[j]:size(1) * splitted[j]:size(2) ~= 128*spl then
           --                break
             --           end
      local pred = model:forward(input)
      pred = torch.reshape(pred, #classes)
      --preds[j] = pred
      --sum = sum + pred
      confusion:add(pred, target)
      --print (confusion)
      end
      --getClass(preds,target,confusion)
      --confusion:add(sum,target)
  -- end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if true then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      --testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end

--print(torch.randperm(11))

for i = 1, 100 do
    train()
    test()
end