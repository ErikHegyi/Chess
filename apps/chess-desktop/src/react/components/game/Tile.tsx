interface Piece {
    piece_type: String;
    player: number;
    first_move: boolean;
}


interface TileProperties {
    key: number;
    piece: Piece | null;
    color: string;
    isHighLighted?: boolean;
    isLastMove?: boolean;
    onClick?: () => void;
}


export const Tile = ({key, piece, color, isHighLighted = false, isLastMove = false, onClick}: TileProperties) => {
    let newColor = isLastMove && isHighLighted && key < 0 ? '#fuckme' : color;
    return (
        <div onClick={onClick} style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            backgroundColor: newColor,
            cursor: 'pointer',
            fontSize: '40px', // Piece size
            userSelect: 'none' // Prevent highlighting text when clicking fast
        }}>
            {piece ? renderPiece(piece) : null}
        </div>
    )
};


// Simple helper to map your Rust piece strings to Unicode or Images
const renderPiece = (piece: Piece) => {
    // Example: if Rust sends "wP", show white pawn unicode
    if ( piece.player === 1 ) {
        switch (piece.piece_type) {
            case 'Pawn': return '♟';
            case 'Bishop': return '♝';
            case 'Knight': return '♞';
            case 'Rook': return '♜';
            case 'Queen': return '♛';
            case 'King': return '♚';
            default: return piece.piece_type.charAt(0)
        }
    }
    else {
        switch (piece.piece_type) {
            case 'Pawn': return '♙';
            case 'Bishop': return '♗';
            case 'Knight': return '♘';
            case 'Rook': return '♖';
            case 'Queen': return '♕';
            case 'King': return '♔';
            default: return piece.piece_type.charAt(0)
        }
    }
};