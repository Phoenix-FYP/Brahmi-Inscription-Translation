import CharacterImage from "./CharacterImage";
import CorpusDisplay from "./CorpusDisplay";

const CharacterSection = ({ title, characterImages, mappedCorpus }) => {
  return (
    <div className="bg-slate-100 rounded-2xl p-6 shadow-md mb-8">
      <h3 className="text-xl font-bold text-gray-700 mb-4 text-center">{title}</h3>

      <div className="flex gap-4 flex-wrap justify-center">
        {characterImages.map((img, idx) => (
          <CharacterImage key={idx} imageUrl={img} />
        ))}
      </div>

      <CorpusDisplay mappedText={mappedCorpus} />
    </div>
  );
};

export default CharacterSection;
