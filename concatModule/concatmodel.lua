require 'nn'
-- A function to create a concat model that concats any models with in models
createConcatModel = function(models)
local model = nn.ConcatTable()
for k,v in pairs(models)
do
    model:add(v)
end
return model
end

-- This Describes the default model to be generate for classification.
DefaultModel = function(num_output)
local mlp=nn.Sequential()
mlp:add(nn.TemporalConvolution(128,128,8,8))
--mlp:add(nn.Reshape(inp))
--mlp:add(nn.Sigmoid())
mlp:add(nn.TemporalMaxPooling(8))

mlp:add(nn.Linear(128,32))
mlp:add(nn.ReLU())
mlp:add(nn.Dropout(.9))
mlp:add(nn.Linear(32,num_output))
mlp:add(nn.LogSoftMax())
--mlp:add(nn.Square())
mlp:add(nn.Sum(1))
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
a = GenerateBagOfClassifiers(4)
b = torch.randn(1000,128)
print(a:forward(b))