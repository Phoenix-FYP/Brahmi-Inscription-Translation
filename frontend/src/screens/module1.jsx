import React, { useState } from 'react';
import { FaUpload } from 'react-icons/fa';
import axios from 'axios';

const Module1 = ({ corpusOutput, setCorpusOutput }) => {
	const [selectedImages, setSelectedImages] = useState([]);
	const [threshold, setThreshold] = useState(2000);
	const [denoisedImages, setDenoisedImages] = useState([]);
	const [segmentedImages, setSegmentedImages] = useState([]);
	const [file, setFile] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState('');

	const handleFileChange = (e) => {
		setFile(e.target.files[0]);
		setError('');
	};

	const handleSelectImage = (img) => {
		setSelectedImages((prev) =>
			prev.includes(img) ? prev.filter((i) => i !== img) : [...prev, img]
		);
	};

	const handleSubmit = async () => {
		if (!file) {
			setError('Please upload an image.');
			return;
		}

		setLoading(true);
		setError('');
		const formData = new FormData();
		formData.append('file', file);
		formData.append('threshold', threshold);

		try {
			console.log("I'm here before sending the request to the backend.");
			const response = await axios.post(
				'http://localhost:8000/api/run-pipeline/',
				formData,
				{
					headers: { 'Content-Type': 'multipart/form-data' },
				}
			);
			console.log('Im here');
			console.log('Response from backend:', response.data);
			// Update state with returned images
			setDenoisedImages([
				`/images/module-1/image_${response.data.image_no}/dilated.png`,
				`/images/module-1/image_${response.data.image_no}/cropped_image.png`,
				`/images/module-1/image_${response.data.image_no}/denoised_image.png`,
				`/images/module-1/image_${response.data.image_no}/denoised_inverted_image.png`,
				`/images/module-1/image_${response.data.image_no}/final_image.png`,
				`/images/module-1/image_${response.data.image_no}/final_image_second_pass.png`,
			]);
			setSegmentedImages(
				(response.data.char_images || []).sort((a, b) => {
					const getCharIndex = (path) => {
						const match = path.match(/character_(\d+)\.png$/);
						return match ? parseInt(match[1], 10) : 0;
					};
					return getCharIndex(a) - getCharIndex(b);
				})
			);

			console.log('Segmented Images:', segmentedImages);
		} catch (err) {
			setError('Error processing the image. Please try again.');
			console.error(err);
		} finally {
			setLoading(false);
		}
	};

	const handleProceed = async () => {
		if (segmentedImages.length === 0) {
			setError('No segmented images to process.');
			return;
		}

		try {
			setLoading(true);
			const formData2 = new FormData();
			formData2.append(
				'image_no',
				segmentedImages[0].split('image_')[1].split('/')[0]
			);
			formData2.append('total_chars', segmentedImages.length);
			// Send segmented images to Module 2 (assuming a new endpoint)
			const response = await axios.post(
				'http://localhost:8000/api/run-module2/',
				formData2,
				{
					headers: { 'Content-Type': 'multipart/form-data' },
				}
			);
			console.log('Module 2 response:', response.data);
			setCorpusOutput(response.data.predictions.final_sequence || '');
			console.log('Combined Corpus:', response.data.predictions.final_sequence);
			// Handle Module 2 results (e.g., update state or navigate to Module 2 screen)
		} catch (err) {
			setError('Error processing Module 2. Please try again.');
			console.error(err);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="m-8">
			<h2 className="text-3xl font-bold text-center mb-8">
				Image Preprocessing Component
			</h2>

			{/* Upload Section */}
			<div className="bg-[#e3d6bf] mb-8 justify-center flex flex-col items-center p-15 max-w-3xl mx-auto rounded-lg shadow-lg">
				<p className="font-semibold mb-4">
					Please upload your Estampage Image:
				</p>

				<label className="w-full max-w-md flex flex-col items-center justify-center border-2 border-dashed border-gray-400 rounded-lg p-6 bg-white cursor-pointer">
					<FaUpload className="text-2xl text-[#D8AF78] mb-2" />
					<span className="text-sm text-gray-600">Click to upload</span>
					<input
						type="file"
						className="hidden"
						accept="image/*"
						onChange={handleFileChange}
					/>
				</label>

				<div className="flex items-center gap-4 mt-6">
					<label htmlFor="threshold" className="font-medium">
						Enter the Threshold Value:
					</label>
					<input
						type="number"
						id="threshold"
						className="px-3 py-1 border border-gray-300 bg-white rounded-md w-32"
						value={threshold}
						onChange={(e) => setThreshold(e.target.value)}
					/>
				</div>
				<p className="text-red-600 font-semibold mt-2 text-sm">
					{error || 'Threshold value should be adjusted accordingly.'}
				</p>

				<div className="text-center mt-4">
					<button
						onClick={handleSubmit}
						disabled={loading}
						className="px-6 py-2 bg-[#a68518] text-white rounded-lg font-semibold hover:bg-[#9b8e59] disabled:bg-gray-400"
					>
						{loading ? 'Processing...' : 'Submit'}
					</button>
				</div>
			</div>

			{/* Denoised Images */}
			{denoisedImages.length > 0 && (
				<div className="bg-[#e3d6bf] mb-8 justify-center flex flex-col items-center p-15 max-w-3xl mx-auto rounded-lg shadow-lg">
					<h3 className="text-xl font-semibold mb-4 text-center">
						Denoised Process Images
					</h3>
					<div className="flex gap-4 flex-wrap justify-center">
						{denoisedImages.map((src, idx) => (
							<img
								key={idx}
								src={src}
								alt="denoised"
								className="w-auto h-48 rounded-md object-cover border border-gray-300"
								onError={() => console.error(`Failed to load image: ${src}`)}
							/>
						))}
					</div>
				</div>
			)}

			{/* Segmented Characters */}
			{segmentedImages.length > 0 && (
				<div className="bg-[#e3d6bf] mb-8 justify-center flex flex-col items-center p-15 max-w-3xl mx-auto rounded-lg shadow-lg">
					<h3 className="text-xl font-semibold mb-2 ">Segmented Characters</h3>
					<p className="text-sm text-gray-600 mb-4 r">
						You can select the image and discard if it's noise.
					</p>
					<div className="flex gap-4 flex-wrap justify-center">
						{segmentedImages.map((src, idx) => (
							<div
								key={idx}
								onClick={() => handleSelectImage(src)}
								className={`p-1 rounded-md border-2 cursor-pointer ${
									selectedImages.includes(src)
										? 'border-'
										: 'border-transparent'
								}`}
							>
								<img
									src={src}
									alt={`char-${idx}`}
									className="w-auto h-32 rounded-md object-cover border border-gray-300"
									onError={() => console.error(`Failed to load image: ${src}`)}
								/>
							</div>
						))}
					</div>
				</div>
			)}

			{/* Proceed Button */}
			{segmentedImages.length > 0 && (
				<div className="text-center">
					<button
						onClick={handleProceed}
						disabled={loading}
						className="px-6 py-2 bg-[#a68518] text-white rounded-lg font-semibold hover:bg-[#9b8e59] disabled:bg-gray-400"
					>
						{loading ? 'Processing...' : 'Proceed to Next Stage'}
					</button>
				</div>
			)}
		</div>
	);
};

export default Module1;
