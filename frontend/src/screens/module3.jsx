import React, { useState, useEffect } from 'react';
import SectionBox from '../components/SectionBox';

const sections = [
	{
		title: 'Input',
		content: '',
	},
	{
		title: 'Bert Based Model Segementation',
		content: '',
	},
	{
		title: 'LTR Greedy Based Segmentation',
		content: '',
	},
	{
		title: 'RTL Greedy Based Segmentation',
		content: '',
	},
	{
		title: 'Calculations',
		content: '',
	},
	{
		title: 'mt5 Model Prediction',
		content: '',
	},
	{
		title: 'Need Review',
		content: '',
	},
];

const Module3 = ({ module3data }) => {
	const [inputOfModel2, setinputtOfModule3] = useState(null);
	const [error, setError] = useState('');
	const [loading, setLoading] = useState(false);
  const [bertModule, setBertModule] = useState('');
  const [bestWords, setBestWords] = useState('');
  const [inputModule, setInputModule] = useState('');
  const [ltrModule, setLtrModule] = useState('');
  const [rtlModule, setRtlModule] = useState('');
  const [needCorrection, setNeedCorrection] = useState('');
	useEffect(() => {
		const fetchModule2Results = async () => {
			if (!module3data) {
				setError('No corpus output provided.');
				return;
			}
			console.log('Module3 component rendered with module3data:', module3data);

			setinputtOfModule3(module3data);
      		console.log(
				'Module3 component rendered with inputModule:',
				inputOfModel2
			);
			console.log('Module3 component rendered with bertModule:', bertModule);


			setBestWords(module3data?.['best']?.['sentence'])
			setInputModule(module3data?.['input'])
			const bertWordsArray = module3data?.['bert'] || [];
      const ltrwordsArray = module3data?.['ltr'] || [];
      const rtlWordsArray = module3data?.['rtl'] || [];
      setLtrModule(ltrwordsArray.join(' ').trim().replace(/\s+/g, ' '));
      setRtlModule(rtlWordsArray.join(' ').trim().replace(/\s+/g, ' '));
			setBertModule(bertWordsArray.join(' ').trim().replace(/\s+/g, ' '))
      const needsCorrection = module3data?.needs_correction;
      const needCorrection = needsCorrection ? "Yes" : "No";
      setNeedCorrection(needCorrection);
      setError('');
			setLoading(false); // Set to false since nothing async is running now
		};

		fetchModule2Results(); // <-- CALL IT HERE
	}, [module3data]);

	const handleGoToStart = () => {
		document.getElementById('module1')?.scrollIntoView({ behavior: 'smooth' });
	};

	return (
		<div className="bg-[#e3d6bf] mb-8 justify-center flex flex-col items-center p-15 max-w-3xl mx-auto rounded-lg shadow-lg">
			<h2 className="text-2xl font-bold mb-6 text-gray-800">Module 3</h2>
			<SectionBox key={1} title={'Input'} content={inputModule} />
			<SectionBox
				key={2}
				title={'Bert Based Model Segementation'}
				content={bertModule}
			/>
			<SectionBox
				key={3}
				title={'LTR Greedy Based Segmentation'}
				content={ltrModule}
			/>
			<SectionBox
				key={4}
				title={'RTL Greedy Based Segmentation'}
				content={rtlModule}
			/>
			<SectionBox
				key={5}
				title={'Best Candidate'}
        content={bestWords}
			/>
			<SectionBox
				key={6}
				title={'Need Correction'}
        content={needCorrection}
			/>
			{/* <SectionBox key={idx} title={"mt5 Model Prediction"} content={inputOfModel2?.["mt5"]} />
      <SectionBox key={idx} title={"Need Review"} content={inputOfModel2?.["need_review"]} /> */}

			<div className="mt-8 text-center">
				<button
					onClick={handleGoToStart}
					className="px-6 py-2 bg-green-600 text-white rounded-xl shadow hover:bg-green-700 transition"
				>
					Go to Start
				</button>
			</div>
		</div>
	);
};

export default Module3;
