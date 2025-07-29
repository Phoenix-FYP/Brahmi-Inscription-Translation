const CharacterImage = ({ imageUrl }) => {
  return (
    <img
      src={imageUrl}
      alt="Character"
      className="w-16 h-16 object-contain border p-1 rounded bg-white shadow"
    />
  );
};

export default CharacterImage;