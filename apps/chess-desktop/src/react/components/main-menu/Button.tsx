import { useState } from 'react';
import '../../../css/Button.css';


interface ButtonProperties {
    text: string;
    onClick?: Function;
    selected?: boolean;
}

export const Button = ({ text, onClick, selected = false }: ButtonProperties) => {
    // Define the state and the handler
    const [isSelected, setIsSelected] = useState<boolean>(selected);

    // Handle clicks
    const handleClick = () => {
        if ( onClick ) { onClick(); }
        setIsSelected(!isSelected);
    };

    // Determine the lass name dynamically
    const className = isSelected ? 'btn selected' : 'btn';

    // Create the button
    return (
        <button className={className} onClick={handleClick}>{text}</button>
    )
}