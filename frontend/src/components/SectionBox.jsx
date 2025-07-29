const SectionBox = ({ title, content }) => {
  return (
    <div className="mb-6 bg-gray-100 p-6 rounded-xl shadow-md min-w-[300px] w-full">
      <h3 className="text-xl font-semibold text-gray-700 mb-3">{title}</h3>
      <div className="bg-white p-4 rounded-lg shadow-sm text-gray-800 whitespace-pre-line">
        {content}
      </div>
    </div>
  );
};

export default SectionBox;
