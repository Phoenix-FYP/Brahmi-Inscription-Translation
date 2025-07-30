import React, { useState, useEffect } from "react";
import axios from "axios";
import CharacterSection from "../components/CharacterSection";

const Module2 = ({ corpusOutput,module3data,setModule3Data }) => {
  const [module2Data, setModule2Data] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const[finalsequence, setFinalSequence] = useState("");
 

  console.log("Module2 component rendered with corpusOutput:", corpusOutput);

  useEffect(() => {
  const fetchModule2Results = async () => {
    if (!corpusOutput) {
      setError("No corpus output provided.");
      return;
    }

    setFinalSequence(corpusOutput);
    setModule2Data(corpusOutput)
    setError("");
    setLoading(false); // Set to false since nothing async is running now
  };

  fetchModule2Results(); // <-- CALL IT HERE
}, [corpusOutput]);


  //     try {
  //       const response = await axios.post(
  //         "http://localhost:8000/api/run-module2/",
  //         {
  //           image_no,
  //           segmented_images: segmentedImages,
  //         },
  //         {
  //           headers: { "Content-Type": "application/json" },
  //         }
  //       );

  //       // Transform backend response to match CharacterSection format
  //       const predictions = response.data.predictions || {};
  //       const formattedData = [
  //         {
  //           title: "Character Recognition Results",
  //           characterImages: segmentedImages,
  //           mappedCorpus: predictions.final_sequence || "No predictions available",
  //         },
  //       ];
  //       setModule2Data(formattedData);
  //     } catch (err) {
  //       setError("Error fetching Module 2 results. Please try again.");
  //       console.error(err);
  //     } finally {
  //       setLoading(false);
  //     }
  //   };

  //   fetchModule2Results();
  // }, [image_no, segmentedImages]);

  // Combine all mappedCorpus strings into one
  // const combinedCorpus = module2Data.map((s) => s.mappedCorpus).join(" ");

const handleProceedToModule3 = async () => {
	if (!finalsequence) {
		setError('No final sequence available to proceed.');
		return;
	}

	try {
		setLoading(true);

		const formData3 = new FormData();
		formData3.append(
      'final_sequence',
      finalsequence
		);

		const response = await axios.post(
			'http://localhost:8000/api/run-module3/',
			formData3,
			{
				headers: { 'Content-Type': 'multipart/form-data' },
			}
		);

		console.log('Module 3 response:', response.data);
    setModule3Data(response.data.predictions || []);

   
		// Assuming Module 3 returns a corrected sequence or similar output
		// setModule3Data(finalsequence);
		// console.log('Grammar Corrected Output:', response.data.corrected_sequence);
		
		// You can also handle additional state updates or navigation here

	} catch (err) {
		setError('Error processing Module 3. Please try again.');
		console.error(err);
	} finally {
		setLoading(false);
	}
};


  return (
    <div className="bg-[#e3d6bf] mb-8 justify-center flex flex-col items-center p-15 max-w-3xl mx-auto rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Module 2</h2>

      {loading && (
        <p className="text-center text-gray-600">Processing Module 2...</p>
      )}

      {error && (
        <p className="text-center text-red-600 font-semibold mb-4">{error}</p>
      )}

      {finalsequence && !error && (
        <p className="text-center text-gray-600 text-4xl font-semibold mb-4">
          {finalsequence}
        </p>
      )}

      {/* {module2Data.map((section, index) => (
        <CharacterSection
          key={index}
          title={section.title}
          characterImages={section.characterImages}
          mappedCorpus={section.mappedCorpus}
        />
      ))} */}

      {module2Data.length > 0 && (
        <div className="mt-6 text-center">
          <button
            onClick={handleProceedToModule3}
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-xl shadow hover:bg-blue-700 transition disabled:bg-gray-400"
          >
            Proceed to Word Segmentation
          </button>
        </div>
      )}
    </div>
  );
};

export default Module2;