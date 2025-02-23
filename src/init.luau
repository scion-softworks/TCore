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

-- Compute the broadcasted shape for two tensor shapes
local function broadcastShapes(shapeA: {number}, shapeB: {number}): {number}
	local lenA = #shapeA
	local lenB = #shapeB

	local maxLen = math.max(lenA, lenB)
	local result = {}

	for i = 1, maxLen do
		local aDim = shapeA[lenA - i + 1] or 1
		local bDim = shapeB[lenB - i + 1] or 1

		if aDim ~= bDim and aDim ~= 1 and bDim ~= 1 then
			error("Shapes are not broadcastable")
		end

		result[maxLen - i + 1] = math.max(aDim, bDim)
	end

	return result
end

-- Convert a linear index to a multi-index for a given shape
local function linearToMultiIndex(linear: number, shape: {number}): {number}
	local multi = {}
	local remainder = linear - 1
	for i = #shape, 1, -1 do
		local idx = (remainder % shape[i]) + 1
		multi[i] = idx
		remainder = math.floor(remainder / shape[i])
	end
	return multi
end

-- Convert a multi-index (array of indices) to a linear index given a shape
local function multiIndexToLinearIndex(multi: {number}, shape: {number}): number
	local index = 0
	local stride = 1
	for i = #shape, 1, -1 do
		index = index + (multi[i] - 1) * stride
		stride = stride * shape[i]
	end
	return index + 1
end

-- Given a broadcast multi-index and the original tensor shape, compute the corresponding linear index for that tensor
local function getTensorIndex(broadcastMulti: {number}, tensorShape: {number}): number
	local tensorMulti = {}
	-- Align dimensions from the right.
	local offset = #broadcastMulti - #tensorShape
	for i = 1, #tensorShape do
		if tensorShape[i] == 1 then
			tensorMulti[i] = 1
		else
			tensorMulti[i] = broadcastMulti[i + offset]
		end
	end
	return multiIndexToLinearIndex(tensorMulti, tensorShape)
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
	-- If one of the inputs is a number, treat it as a scalar and broadcast it.
	if type(a) == "number" or type(b) == "number" then
		local baseTensor = if (type(a) == "number" and type(b) == 'table') then (b :: Tensor) else (a :: Tensor)
		local result = create(baseTensor.shape, 0)
		for i = 1, baseTensor.shapeProduct do
			local aElement = if (type(a) == "number") then a else readElement((a :: Tensor), i)
			local bElement = if (type(b) == "number") then b else readElement((b :: Tensor), i)
			writeElement(result, i, operation(aElement, bElement))
		end
		return result
	end

	local tensorA = a :: Tensor
	local tensorB = b :: Tensor

	local broadcastShape = broadcastShapes(tensorA.shape, tensorB.shape)
	local broadcastProduct = getShapeProduct(broadcastShape)
	local result = create(broadcastShape, 0)

	for i = 1, broadcastProduct do
		local multi = linearToMultiIndex(i, broadcastShape)
		local indexA = getTensorIndex(multi, tensorA.shape)
		local indexB = getTensorIndex(multi, tensorB.shape)
		local aElement = readElement(tensorA, indexA)
		local bElement = readElement(tensorB, indexB)
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

local function sum(a: Tensor)
	local total = 0

	for n = 1, a.shapeProduct do
		total += readElement(a, n)
	end

	return total
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

local function transpose(a: Tensor)
	assert(#a.shape == 2, "Transpose only supports 2D tensors.")

	local m, n = a.shape[1], a.shape[2]
	local result = create({n, m}, 0)

	for i = 1, m do
		for j = 1, n do
			local val = readElement(a, (i - 1) * n + j)
			writeElement(result, (j - 1) * m + i, val)
		end
	end

	return result
end

local function toNestedTable(a: Tensor): {any}
    local currentIndex = 1

    local function fillTable(shape: TensorShape, dim: number): any
        if dim > #shape then
            local value = readElement(a, currentIndex)
            currentIndex = currentIndex + 1
            return value
        end

        local result = {}
        for i = 1, shape[dim] do
            table.insert(result, fillTable(shape, dim + 1))
        end
        return result
    end

    return fillTable(a.shape, 1)
end

local function toFlatArray(a: Tensor): {number}
	local result = {}

	for i = 1, a.shapeProduct do
		table.insert(result, readElement(a, i))
	end

	return result
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
Tensor.sum = sum
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
Tensor.transpose = transpose
Tensor.toFlatArray = toFlatArray
Tensor.toNestedTable = toNestedTable
Tensor.elementWiseOperation = elementWiseOperation

return Tensor