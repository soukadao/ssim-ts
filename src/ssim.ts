import Decimal from "decimal.js";

const K1 = new Decimal(0.01);
const K2 = new Decimal(0.03);
const L = new Decimal(255); // L the dynamic range of the pixel-values

/**
 * Sample Mean
 * @param x
 * @returns
 */
function sampleMean(x: Uint8ClampedArray): Decimal {
	let sum = new Decimal(0);

	for (let i = 0; i < x.length; i++) {
		sum = sum.add(x[i]);
	}

	return sum.div(x.length);
}

/**
 * Sample Variance
 * @param x
 * @returns
 */
function sampleVariance(x: Uint8ClampedArray): Decimal {
	const pixelSampleMean = sampleMean(x);
	let sum = new Decimal(0);

	for (let i = 0; i < x.length; i++) {
		const diff = new Decimal(x[i]).sub(pixelSampleMean);
		sum = sum.add(diff.pow(2));
	}

	return sum.div(x.length);
}

/**
 * Sample Covariance
 * @param x
 * @param y
 * @returns
 */
function sampleCovariance(x: Uint8ClampedArray, y: Uint8ClampedArray): Decimal {
	const meanX = sampleMean(x);
	const meanY = sampleMean(y);
	let sum = new Decimal(0);

	for (let i = 0; i < x.length; i++) {
		const diffX = new Decimal(x[i]).sub(meanX);
		const diffY = new Decimal(y[i]).sub(meanY);
		sum = sum.add(diffX.mul(diffY));
	}

	return sum.div(x.length);
}

/**
 * Calculate SSIM between two images
 * @param x First image
 * @param y Second image
 * @returns SSIM value between -1 and 1 (1 means identical)
 */
export function ssim(x: ImageData, y: ImageData): number {
	if (x.width !== y.width || x.height !== y.height) {
		throw new Error("Images must have the same dimensions");
	}

	const c1 = K1.mul(L).pow(2);
	const c2 = K2.mul(L).pow(2);

	const xData = x.data;
	const yData = y.data;

	const meanX = sampleMean(xData);
	const meanY = sampleMean(yData);
	const varianceX = sampleVariance(xData);
	const varianceY = sampleVariance(yData);
	const covarianceXY = sampleCovariance(xData, yData);

	const numerator1 = meanX.mul(meanY).mul(2).add(c1);
	const denominator1 = meanX.pow(2).add(meanY.pow(2)).add(c1);
	const luminance = numerator1.div(denominator1);

	const sqrtVarianceX = varianceX.sqrt();
	const sqrtVarianceY = varianceY.sqrt();
	const numerator2 = sqrtVarianceX.mul(sqrtVarianceY).mul(2).add(c2);
	const denominator2 = varianceX.add(varianceY).add(c2);
	const contrast = numerator2.div(denominator2);

	const numerator3 = covarianceXY.add(c2.div(2));
	const denominator3 = sqrtVarianceX.mul(sqrtVarianceY).add(c2.div(2));
	const structure = numerator3.div(denominator3);

	return luminance.mul(contrast).mul(structure).toNumber();
}
