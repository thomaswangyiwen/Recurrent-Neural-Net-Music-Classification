require 'nn'
-- A function to create a concat model that concats any models with in models
createConcatModel = function(models)
-- Create model
local model = nn.Sequential()
-- Create concat table 
local concat = nn.ConcatTable()
for k,v in pairs(models)
do
    -- concat models to concat table
    concat:add(v)
end

-- concat concat to model 
model:add(concat)

-- Bring all outputs together Join the Dark Side.
model:add(nn.JoinTable(1))
return model
end

-- This Describes the default model to be generate for classification.
DefaultModel = function(num_output)
local mlp=nn.Sequential()
mlp:add(nn.TemporalConvolution(128,128,8,8))
mlp:add(nn.Tanh())
mlp:add(nn.TemporalMaxPooling(8))
mlp:add(nn.Linear(128,64))
--mlp:add(nn.Dropout())
mlp:add(nn.SoftSign())
mlp:add(nn.Linear(64,num_output))
mlp:add(nn.Sum())
mlp:add(nn.Sigmoid())

return mlp
end


-- Generating a bag of classifiers -- of default model type
GenerateBagOfClassifiers = function(numberofclasses)
local models = {}
for i=1,numberofclasses
	do
	-- This is for the 1,-1 class case 
	models[i] = DefaultModel(1)
	-- This is for the yes or no case (the network has 2 outputs)
	--models[i] = DefaultModel(2)
end

return createConcatModel(models)

end

--- Here is an example of how these functions can be used.
local a = GenerateBagOfClassifiers(4)
local b = torch.randn(1000,128)
local m = DefaultModel(1)
local m2 = nn.Linear(128,4)
print(a:forward(b))
print(m:forward(b))
--print(m2:forward(b))