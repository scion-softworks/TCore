--!native
--!optimize 2
local RNG = Random.new()
local Tensor = {}

export type TensorShape = {number}
export type Tensor = {shape: TensorShape, data: buffer, shapeProduct: number}

local writeu8 = buffer.writeu8
local writeu16 = buffer.writeu16
local writeu32 = buffer.writeu32
local writei8 = buffer.writei8
local writei16 = buffer.writei16
local writei32 = buffer.writei32
local writef32 = buffer.writef32
local writef64 = buffer.writef64

local readu8 = buffer.readu8
local readu16 = buffer.readu16
local readu32 = buffer.readu32
local readi8 = buffer.readi8
local readi16 = buffer.readi16
local readi32 = buffer.readi32
local readf32 = buffer.readf32
local readf64 = buffer.readf64

local function getShapeProduct(shape: TensorShape)
	local shapeProduct = 1

	for _, dim in shape do shapeProduct *= dim end

	return shapeProduct
end

local function readElement(tensor: Tensor, index: number): number
	return readf64(tensor.data, (index-1)*8)
end

local function writeElement(tensor: Tensor, index: number, value: number)
	writef64(tensor.data, (index-1)*8, value)
end

local function create(shape: TensorShape, fillValue: number | (number) -> number): Tensor
	local shapeProduct = getShapeProduct(shape)
	
	local bufferSize = (shapeProduct * 8)
	if bufferSize >= 1073741824 then 
		error(`Buffer size ({bufferSize} bytes) exceeds maximum allowed (1 GiB or 1073741824 bytes)'. Adjust your shape to be below the limit`)
	end
	
	local success, buf = pcall(buffer.create, bufferSize)

	if not success then error('Failed to create tensor. The device might be running low on memory, try to lower the overall tensor shape') end

	local fillValueType = type(fillValue)

	if fillValueType == 'number' then
		for i = 0, bufferSize-8, 8 do
			writef64(buf, i, fillValue :: number)
		end
	else
		for i = 0, bufferSize-8, 8 do
			writef64(buf, i, (fillValue :: (number) -> number)(i))
		end
	end

	return {
		shape = shape; 
		data = buf; 
		shapeProduct = shapeProduct
	}
end

local function elementWiseOperation(a: Tensor | number, b: Tensor | number, operation: (a: number, b: number) -> number)
	local baseTensor

	if type(a) == 'number' and type(b) ~= 'number' then
		baseTensor = b
	elseif type(b) == 'number' and type(a) ~= 'number' then
		baseTensor = a
	elseif type(a) == 'number' and type(b) == 'number' then
		error('Cannot perform element-wise operation on two numbers')
	elseif type(a) == 'table' and type(b) == 'table' then
		local aDimCount = #(a :: Tensor).shape
		local bDimCount = #(b :: Tensor).shape

		-- Check that shapes are identical
		if aDimCount ~= bDimCount then
			error('Shape mismatch: cannot operate on tensors with different number of dimensions.')
		end

		for i = 1, aDimCount do
			if (a :: Tensor).shape[i] ~= (b :: Tensor).shape[i] then
				error('Shape mismatch: full broadcasting not implemented.')
			end
		end
		baseTensor = a
	end
  
	local result = create(baseTensor.shape, 0)

	for i = 1, baseTensor.shapeProduct do
		local aElement = if (type(a) == 'number') then a else readElement(a, i)
		local bElement = if (type(b) == 'number') then b else readElement(b, i)

		writeElement(result, i, operation(aElement, bElement))
	end

	return result
end

local function zeros(shape: TensorShape)
	return create(shape, 0)
end

local function ones(shape: TensorShape)
	return create(shape, 1)
end

local function randInt(shape: TensorShape, min: number?, max: number?)
	min, max = min or 0, max or 1
	return create(shape, function() return RNG:NextInteger(min :: number, max :: number) end)
end

local function randNum(shape: TensorShape, min: number?, max: number?)
	min, max = min or 0, max or 1
	return create(shape, function() return RNG:NextNumber(min :: number, max :: number) end)
end

local function add(a, b)
	return elementWiseOperation(a, b, function(a, b) return a + b end)
end

local function subtract(a, b)
	return elementWiseOperation(a, b, function(a, b) return a - b end)
end

local function multiply(a, b)
	return elementWiseOperation(a, b, function(a, b) return a * b end)
end

local function divide(a, b)
	return elementWiseOperation(a, b, function(a, b) return a / b end)
end

local function scale(a, scalar)
	return elementWiseOperation(a, scalar, function(a, s) return a * s end)
end

local function matMul(a, b)
	if #a.shape ~= 2 or #b.shape ~= 2 then
	  	error('matMul only supports 2D tensors (matrices)')
	end

	local m, n = a.shape[1], a.shape[2]
	local nB, p = b.shape[1], b.shape[2]

	if n ~= nB then error('Inner dimensions must match for matrix multiplication') end

	local result = create({m, p}, 0)

	for i = 1, m do
		for j = 1, p do
			local sum = 0
			for k = 1, n do
				local aElement = readElement(a, (i-1)*n + k)
				local bElement = readElement(b, (k-1)*p + j)

				sum += aElement * bElement
			end
			writeElement(result, (i-1)*p + j, sum)
		end
	end

	return result
end

local function reshape(a: Tensor, newShape: TensorShape)
	if getShapeProduct(newShape) ~= a.shapeProduct then
	  	error('New shape must have the same total number of elements as the original tensor.')
	end

	return { shape = newShape, shapeProduct = getShapeProduct(newShape), data = a.data }
end

local function flatten(a: Tensor)
	return reshape(a, {a.shapeProduct})
end

local function mean(a: Tensor)
	local total = 0
	local n = a.shapeProduct

	for i = 1, n do
		total += readElement(a, i)
	end

	return total / n
end

local function variance(a: Tensor)
	local m = mean(a)
	local sumSq = 0

	local n = a.shapeProduct

	for i = 1, n do
		local diff = readElement(a, i) - m
		sumSq += diff * diff
	end

	return sumSq / n
end

local function std(a: Tensor)
	local var = variance(a)
	return math.sqrt(var)
end

local function minMaxNormalize(a: Tensor)
	local n = a.shapeProduct

	local minVal = readElement(a, 1)
	local maxVal = minVal

	for i = 2, n do
		local v = readElement(a, i)
		if v < minVal then minVal = v end
		if v > maxVal then maxVal = v end
	end

	local range = maxVal - minVal

	if range == 0 then range = 1 end

	return elementWiseOperation(a, 0, function(a, _)
	  	return (a - minVal) / range
	end)
end
  
local function zScoreNormalize(a: Tensor)
	local m = mean(a)
	local s = std(a)

	if s == 0 then s = 1 end

	return elementWiseOperation(a, 0, function(a, _)
	  	return (a - m) / s
	end)
end

local function argmax(a: Tensor): (number, number)
	if a.shapeProduct == 0 then 
		error('Tensor has no elements')
	end

	local bestIndex = 1
	local bestVal = readElement(a, 1)
  
	for i = 2, a.shapeProduct do
		local v = readElement(a, i)
		if v > bestVal then
			bestVal = v
			bestIndex = i
		end
	end
  
	return bestIndex, bestVal
end

local function printTensor(a: Tensor)
	local parts = {}
	table.insert(parts, 'Tensor(shape: {')
	for i, dim in ipairs(a.shape) do
		table.insert(parts, tostring(dim))
		if i < #a.shape then
			table.insert(parts, ', ')
		end
	end
	table.insert(parts, '}, data: [')
	for i = 1, a.shapeProduct do
		table.insert(parts, tostring(readElement(a, i)))
		if i < a.shapeProduct then
			table.insert(parts, ', ')
		end
	end
	table.insert(parts, '])')
	
	print(table.concat(parts))
end

Tensor.readElement = readElement
Tensor.writeElement = writeElement
Tensor.create = create
Tensor.zeros = zeros
Tensor.ones = ones
Tensor.randInt = randInt
Tensor.randNum = randNum
Tensor.add = add
Tensor.subtract = subtract
Tensor.multiply = multiply
Tensor.divide = divide
Tensor.scale = scale
Tensor.matMul = matMul
Tensor.reshape = reshape
Tensor.flatten = flatten
Tensor.mean = mean
Tensor.variance = variance
Tensor.std = std
Tensor.minMaxNormalize = minMaxNormalize
Tensor.zScoreNormalize = zScoreNormalize
Tensor.argmax = argmax
Tensor.print = printTensor

return Tensor