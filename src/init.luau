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

function Tensor.zeros(shape: TensorShape)
	return Tensor.create(shape, 0)
end

function Tensor.ones(shape: TensorShape)
	return Tensor.create(shape, 1)
end

function Tensor.randInt(shape: TensorShape, min: number?, max: number?)
	min, max = min or 0, max or 1
	return Tensor.create(shape, function() return RNG:NextInteger(min :: number, max :: number) end)
end

function Tensor.randNum(shape: TensorShape, min: number?, max: number?)
	min, max = min or 0, max or 1
	return Tensor.create(shape, function() return RNG:NextNumber(min :: number, max :: number) end)
end

function Tensor.readElement(tensor: Tensor, index: number): number
	return readf64(tensor.data, (index-1)*8)
end

function Tensor.writeElement(tensor: Tensor, index: number, value: number)
	writef64(tensor.data, (index-1)*8, value)
end

function Tensor.create(shape: TensorShape, fillValue: number | (number) -> number): Tensor
	local shapeProduct = getShapeProduct(shape)
	
	local bufferSize = (shapeProduct * 8)
	if bufferSize >= 1073741824 then 
		error(`Buffer size ({bufferSize} bytes) exceeds maximum allowed (1 GiB or 1073741824 bytes)". Adjust your shape to be below the limit`)
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

function Tensor.elementWiseOperation(a, b, operation: (a: number, b: number) -> number)
	local baseTensor

	if type(a) == "number" then
		baseTensor = b
	elseif type(b) == "number" then
		baseTensor = a
	else
		local aDimCount = #a.shape
		local bDimCount = #b.shape

		-- Check that shapes are identical
		if aDimCount ~= bDimCount then
			error("Shape mismatch: cannot operate on tensors with different number of dimensions.")
		end

		for i = 1, aDimCount do
			if aDimCount ~= bDimCount then
				error("Shape mismatch: full broadcasting not implemented.")
			end
		end
		baseTensor = a
	end
  
	local result = Tensor.create(baseTensor.shape, 0)

	-- !!! parallelize
	for i = 1, baseTensor.shapeProduct do
		local aElement = if (type(a) == "number") then a else Tensor.readElement(a, i)
		local bElement = if (type(b) == "number") then b else Tensor.readElement(b, i)

		Tensor.writeElement(result, i, operation(aElement, bElement))
	end

	return result
end

function Tensor.add(a, b)
	return Tensor.elementWiseOperation(a, b, function(a, b) return a + b end)
end

function Tensor.subtract(a, b)
	return Tensor.elementWiseOperation(a, b, function(a, b) return a - b end)
end

function Tensor.multiply(a, b)
	return Tensor.elementWiseOperation(a, b, function(a, b) return a * b end)
end

function Tensor.divide(a, b)
	return Tensor.elementWiseOperation(a, b, function(a, b) return a / b end)
end

function Tensor.scale(a, scalar)
	return Tensor.elementWiseOperation(a, scalar, function(a, s) return a * s end)
end

function Tensor.matMul(a, b)
	if #a.shape ~= 2 or #b.shape ~= 2 then
	  	error("Tensor.matMul only supports 2D tensors (matrices)")
	end

	local m, n = a.shape[1], a.shape[2]
	local nB, p = b.shape[1], b.shape[2]

	if n ~= nB then error("Inner dimensions must match for matrix multiplication") end

	local result = Tensor.create({m, p}, 0)

	-- !!! parallelize
	for i = 1, m do
		for j = 1, p do
			local sum = 0
			for k = 1, n do
				local aElement = Tensor.readElement(a, (i-1)*n + k)
				local bElement = Tensor.readElement(b, (k-1)*p + j)

				sum += aElement * bElement
			end
			Tensor.writeElement(result, (i-1)*p + j, sum)
		end
	end

	return result
end

return Tensor