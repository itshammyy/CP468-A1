// Main Component

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';

type Player = 'X' | 'O';
type CellValue = Player | null;
type Board = CellValue[][];
type GameAgent = 'Human' | 'Minimax' | 'Alpha-Beta' | 'Expectiminimax' | 'Gemini LLM';

interface GameState {
  board: Board;
  currentPlayer: Player;
  gameActive: boolean;
  winner: Player | 'Draw' | null;
  winningLine: number[][] | null;
  players: { X: GameAgent; O: GameAgent };
}

// AI Algorithm implementations
class GameAI {
  // Minimax Algorithm
  static minimax(board: Board, depth: number, isMaximizing: boolean, player: Player): number {
    const opponent = player === 'X' ? 'O' : 'X';
    const winner = this.checkWinner(board);
    
    if (winner === player) return 10 - depth;
    if (winner === opponent) return depth - 10;
    if (this.isBoardFull(board)) return 0;
    
    if (isMaximizing) {
      let bestScore = -Infinity;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (board[i][j] === null) {
            board[i][j] = player;
            const score = this.minimax(board, depth + 1, false, player);
            board[i][j] = null;
            bestScore = Math.max(score, bestScore);
          }
        }
      }
      return bestScore;
    } else {
      let bestScore = Infinity;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (board[i][j] === null) {
            board[i][j] = opponent;
            const score = this.minimax(board, depth + 1, true, player);
            board[i][j] = null;
            bestScore = Math.min(score, bestScore);
          }
        }
      }
      return bestScore;
    }
  }

  // Alpha-Beta Pruning Algorithm
  static alphaBeta(board: Board, depth: number, alpha: number, beta: number, isMaximizing: boolean, player: Player): number {
    const opponent = player === 'X' ? 'O' : 'X';
    const winner = this.checkWinner(board);
    
    if (winner === player) return 10 - depth;
    if (winner === opponent) return depth - 10;
    if (this.isBoardFull(board)) return 0;
    
    if (isMaximizing) {
      let bestScore = -Infinity;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (board[i][j] === null) {
            board[i][j] = player;
            const score = this.alphaBeta(board, depth + 1, alpha, beta, false, player);
            board[i][j] = null;
            bestScore = Math.max(score, bestScore);
            alpha = Math.max(alpha, bestScore);
            if (beta <= alpha) break; // Alpha-Beta pruning
          }
        }
        if (beta <= alpha) break;
      }
      return bestScore;
    } else {
      let bestScore = Infinity;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (board[i][j] === null) {
            board[i][j] = opponent;
            const score = this.alphaBeta(board, depth + 1, alpha, beta, true, player);
            board[i][j] = null;
            bestScore = Math.min(score, bestScore);
            beta = Math.min(beta, bestScore);
            if (beta <= alpha) break; // Alpha-Beta pruning
          }
        }
        if (beta <= alpha) break;
      }
      return bestScore;
    }
  }

  // Expectiminimax Algorithm (with chance nodes for suboptimal opponents)
  static expectiminimax(board: Board, depth: number, isMaximizing: boolean, player: Player): number {
    const opponent = player === 'X' ? 'O' : 'X';
    const winner = this.checkWinner(board);
    
    if (winner === player) return 10 - depth;
    if (winner === opponent) return depth - 10;
    if (this.isBoardFull(board)) return 0;
    
    if (isMaximizing) {
      let bestScore = -Infinity;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (board[i][j] === null) {
            board[i][j] = player;
            const score = this.expectiminimax(board, depth + 1, false, player);
            board[i][j] = null;
            bestScore = Math.max(score, bestScore);
          }
        }
      }
      return bestScore;
    } else {
      // Chance node: 80% optimal play, 20% random play
      const availableMoves = this.getAvailableMoves(board);
      let optimalScore = Infinity;
      let totalRandomScore = 0;
      
      // Calculate optimal score (80% weight)
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (board[i][j] === null) {
            board[i][j] = opponent;
            const score = this.expectiminimax(board, depth + 1, true, player);
            board[i][j] = null;
            optimalScore = Math.min(score, optimalScore);
          }
        }
      }
      
      // Calculate average random score (20% weight)
      for (const [i, j] of availableMoves) {
        board[i][j] = opponent;
        const score = this.expectiminimax(board, depth + 1, true, player);
        board[i][j] = null;
        totalRandomScore += score;
      }
      
      const averageRandomScore = totalRandomScore / availableMoves.length;
      return 0.8 * optimalScore + 0.2 * averageRandomScore;
    }
  }

  // Get best move for each algorithm
  static getMinimaxMove(board: Board, player: Player): [number, number] {
    let bestMove: [number, number] = [-1, -1];
    let bestScore = -Infinity;
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (board[i][j] === null) {
          board[i][j] = player;
          const score = this.minimax(board, 0, false, player);
          board[i][j] = null;
          
          if (score > bestScore) {
            bestScore = score;
            bestMove = [i, j];
          }
        }
      }
    }
    
    return bestMove;
  }

  static getAlphaBetaMove(board: Board, player: Player): [number, number] {
    let bestMove: [number, number] = [-1, -1];
    let bestScore = -Infinity;
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (board[i][j] === null) {
          board[i][j] = player;
          const score = this.alphaBeta(board, 0, -Infinity, Infinity, false, player);
          board[i][j] = null;
          
          if (score > bestScore) {
            bestScore = score;
            bestMove = [i, j];
          }
        }
      }
    }
    
    return bestMove;
  }

  static getExpectiminimaxMove(board: Board, player: Player): [number, number] {
    let bestMove: [number, number] = [-1, -1];
    let bestScore = -Infinity;
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (board[i][j] === null) {
          board[i][j] = player;
          const score = this.expectiminimax(board, 0, false, player);
          board[i][j] = null;
          
          if (score > bestScore) {
            bestScore = score;
            bestMove = [i, j];
          }
        }
      }
    }
    
    return bestMove;
  }

  // Gemini LLM API integration
  static async getGeminiMove(board: Board, player: Player): Promise<[number, number]> {
    try {
      const prompt = `You are playing Tic-Tac-Toe. You are player ${player}. 
      
Current board state (0=empty, X=X, O=O):
${board.map((row, i) => 
  row.map((cell, j) => cell || `(${i},${j})`).join(' | ')
).join('\n')}

Rules:
- You are ${player}
- Choose the best move to win or block opponent
- Return ONLY the coordinates as [row,col] where row and col are 0, 1, or 2
- Example response: [1,2]

Your move:`;

      // Note: Replace 'YOUR_API_KEY_HERE' with actual OpenRouter API key
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer YOUR_API_KEY_HERE',
          'Content-Type': 'application/json',
          'HTTP-Referer': window.location.origin,
          'X-Title': 'AI Tic-Tac-Toe'
        },
        body: JSON.stringify({
          model: 'google/gemini-pro',
          messages: [
            {
              role: 'user',
              content: prompt
            }
          ],
          max_tokens: 50,
          temperature: 0.1
        })
      });

      if (!response.ok) {
        throw new Error(`API call failed: ${response.status}`);
      }

      const data = await response.json();
      const content = data.choices[0]?.message?.content || '';
      
      // Parse the response to extract coordinates
      const match = content.match(/\[(\d),(\d)\]/);
      if (match) {
        const row = parseInt(match[1]);
        const col = parseInt(match[2]);
        if (row >= 0 && row <= 2 && col >= 0 && col <= 2 && board[row][col] === null) {
          return [row, col];
        }
      }
      
      // Fallback to Minimax if parsing fails
      throw new Error('Invalid API response');
    } catch (error) {
      console.warn('Gemini API failed, falling back to Minimax:', error);
      return this.getMinimaxMove(board, player);
    }
  }

  // Utility functions
  static checkWinner(board: Board): Player | 'Draw' | null {
    // Check rows
    for (let i = 0; i < 3; i++) {
      if (board[i][0] && board[i][0] === board[i][1] && board[i][1] === board[i][2]) {
        return board[i][0];
      }
    }
    
    // Check columns
    for (let j = 0; j < 3; j++) {
      if (board[0][j] && board[0][j] === board[1][j] && board[1][j] === board[2][j]) {
        return board[0][j];
      }
    }
    
    // Check diagonals
    if (board[0][0] && board[0][0] === board[1][1] && board[1][1] === board[2][2]) {
      return board[0][0];
    }
    if (board[0][2] && board[0][2] === board[1][1] && board[1][1] === board[2][0]) {
      return board[0][2];
    }
    
    // Check for draw
    if (this.isBoardFull(board)) {
      return 'Draw';
    }
    
    return null;
  }

  static getWinningLine(board: Board): number[][] | null {
    // Check rows
    for (let i = 0; i < 3; i++) {
      if (board[i][0] && board[i][0] === board[i][1] && board[i][1] === board[i][2]) {
        return [[i, 0], [i, 1], [i, 2]];
      }
    }
    
    // Check columns
    for (let j = 0; j < 3; j++) {
      if (board[0][j] && board[0][j] === board[1][j] && board[1][j] === board[2][j]) {
        return [[0, j], [1, j], [2, j]];
      }
    }
    
    // Check diagonals
    if (board[0][0] && board[0][0] === board[1][1] && board[1][1] === board[2][2]) {
      return [[0, 0], [1, 1], [2, 2]];
    }
    if (board[0][2] && board[0][2] === board[1][1] && board[1][1] === board[2][0]) {
      return [[0, 2], [1, 1], [2, 0]];
    }
    
    return null;
  }

  static isBoardFull(board: Board): boolean {
    return board.every(row => row.every(cell => cell !== null));
  }

  static getAvailableMoves(board: Board): [number, number][] {
    const moves: [number, number][] = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (board[i][j] === null) {
          moves.push([i, j]);
        }
      }
    }
    return moves;
  }
}

const TicTacToeGame: React.FC = () => {
  const { toast } = useToast();
  
  const [gameState, setGameState] = useState<GameState>({
    board: Array(3).fill(null).map(() => Array(3).fill(null)),
    currentPlayer: 'X',
    gameActive: false,
    winner: null,
    winningLine: null,
    players: { X: 'Human', O: 'Minimax' }
  });

  const [isThinking, setIsThinking] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Select players and start the game');

  // Initialize new game
  const initializeGame = useCallback(() => {
    setGameState(prev => ({
      ...prev,
      board: Array(3).fill(null).map(() => Array(3).fill(null)),
      currentPlayer: 'X',
      gameActive: true,
      winner: null,
      winningLine: null
    }));
    setStatusMessage('Game started! Player X\'s turn');
  }, []);

  // Make a move on the board
  const makeMove = useCallback(async (row: number, col: number, player: Player) => {
    setGameState(prev => {
      const newBoard = prev.board.map(r => [...r]);
      newBoard[row][col] = player;
      
      const winner = GameAI.checkWinner(newBoard);
      const winningLine = GameAI.getWinningLine(newBoard);
      
      return {
        ...prev,
        board: newBoard,
        currentPlayer: player === 'X' ? 'O' : 'X',
        winner,
        winningLine,
        gameActive: !winner
      };
    });
  }, []);

  // Get AI move based on selected algorithm
  const getAIMove = useCallback(async (player: Player, board: Board): Promise<[number, number]> => {
    const algorithm = gameState.players[player];
    
    switch (algorithm) {
      case 'Minimax':
        return GameAI.getMinimaxMove(board, player);
      case 'Alpha-Beta':
        return GameAI.getAlphaBetaMove(board, player);
      case 'Expectiminimax':
        return GameAI.getExpectiminimaxMove(board, player);
      case 'Gemini LLM':
        return await GameAI.getGeminiMove(board, player);
      default:
        return GameAI.getMinimaxMove(board, player);
    }
  }, [gameState.players]);

  // Handle turn logic
  const handleTurn = useCallback(async () => {
    if (!gameState.gameActive || gameState.winner) return;
    
    const currentAgent = gameState.players[gameState.currentPlayer];
    
    if (currentAgent !== 'Human') {
      setIsThinking(true);
      setStatusMessage(`Player ${gameState.currentPlayer} (${currentAgent}) is thinking...`);
      
      try {
        const [row, col] = await getAIMove(gameState.currentPlayer, gameState.board);
        
        // Add a small delay to show AI is "thinking"
        setTimeout(() => {
          makeMove(row, col, gameState.currentPlayer);
          setIsThinking(false);
        }, 500);
      } catch (error) {
        console.error('AI move failed:', error);
        setIsThinking(false);
        toast({
          title: 'AI Error',
          description: 'AI failed to make a move. Please try again.',
          variant: 'destructive'
        });
      }
    } else {
      setStatusMessage(`Player ${gameState.currentPlayer}'s turn`);
    }
  }, [gameState, getAIMove, makeMove, toast]);

  // Handle cell click for human players
  const handleCellClick = useCallback((row: number, col: number) => {
    if (!gameState.gameActive || 
        gameState.board[row][col] !== null || 
        gameState.players[gameState.currentPlayer] !== 'Human' ||
        isThinking) {
      return;
    }
    
    makeMove(row, col, gameState.currentPlayer);
  }, [gameState, makeMove, isThinking]);

  // Update status message based on game state
  useEffect(() => {
    if (gameState.winner) {
      if (gameState.winner === 'Draw') {
        setStatusMessage('It\'s a Draw!');
      } else {
        const winnerAgent = gameState.players[gameState.winner];
        setStatusMessage(`Player ${gameState.winner} (${winnerAgent}) Wins!`);
      }
    } else if (gameState.gameActive) {
      handleTurn();
    }
  }, [gameState.currentPlayer, gameState.winner, gameState.gameActive, handleTurn]);

  // Handle player selection
  const handlePlayerChange = useCallback((player: 'X' | 'O', agent: GameAgent) => {
    setGameState(prev => ({
      ...prev,
      players: { ...prev.players, [player]: agent }
    }));
  }, []);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl shadow-lg">
        <CardHeader className="text-center">
          <CardTitle className="text-4xl font-bold text-foreground mb-8">
            AI Tic-Tac-Toe
          </CardTitle>
          
          {/* Player Selection Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="space-y-3">
              <h2 className="text-xl font-semibold text-player-x">Player X</h2>
              <Select
                value={gameState.players.X}
                onValueChange={(value: GameAgent) => handlePlayerChange('X', value)}
                disabled={gameState.gameActive}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Human">Human</SelectItem>
                  <SelectItem value="Minimax">Minimax</SelectItem>
                  <SelectItem value="Alpha-Beta">Alpha-Beta</SelectItem>
                  <SelectItem value="Expectiminimax">Expectiminimax</SelectItem>
                  <SelectItem value="Gemini LLM">Gemini LLM</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-3">
              <h2 className="text-xl font-semibold text-player-o">Player O</h2>
              <Select
                value={gameState.players.O}
                onValueChange={(value: GameAgent) => handlePlayerChange('O', value)}
                disabled={gameState.gameActive}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Human">Human</SelectItem>
                  <SelectItem value="Minimax">Minimax</SelectItem>
                  <SelectItem value="Alpha-Beta">Alpha-Beta</SelectItem>
                  <SelectItem value="Expectiminimax">Expectiminimax</SelectItem>
                  <SelectItem value="Gemini LLM">Gemini LLM</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Game Control Button */}
          <Button
            onClick={initializeGame}
            className="w-full md:w-auto px-8 py-3 text-lg font-semibold"
            disabled={isThinking}
          >
            {gameState.gameActive ? 'Reset Game' : 'Start Game'}
          </Button>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Status Display */}
          <div className="text-center">
            <p className="text-lg font-medium text-foreground">
              {statusMessage}
            </p>
          </div>

          {/* Game Board */}
          <div 
            className="grid grid-cols-3 gap-2 max-w-md mx-auto bg-board-bg p-4 rounded-lg shadow-inner"
            style={{ aspectRatio: '1' }}
          >
            {gameState.board.map((row, i) =>
              row.map((cell, j) => {
                const isWinningCell = gameState.winningLine?.some(([wi, wj]) => wi === i && wj === j);
                return (
                  <button
                    key={`${i}-${j}`}
                    className={`
                      aspect-square border-2 border-cell-border rounded-lg text-4xl font-bold
                      transition-all duration-200 hover:bg-muted/50 focus:outline-none focus:ring-2 focus:ring-ring
                      ${isWinningCell ? 'bg-winning-highlight' : 'bg-card'}
                      ${gameState.players[gameState.currentPlayer] === 'Human' && !cell && gameState.gameActive && !isThinking
                        ? 'cursor-pointer hover:scale-105' 
                        : 'cursor-default'
                      }
                    `}
                    onClick={() => handleCellClick(i, j)}
                    disabled={!gameState.gameActive || !!cell || isThinking || gameState.players[gameState.currentPlayer] !== 'Human'}
                  >
                    {cell && (
                      <span 
                        className={cell === 'X' ? 'text-player-x' : 'text-player-o'}
                      >
                        {cell}
                      </span>
                    )}
                  </button>
                );
              })
            )}
          </div>

          {/* Game Info */}
          <div className="text-center text-sm text-muted-foreground space-y-1">
            <p>
              X: <span className="text-player-x font-medium">{gameState.players.X}</span> | 
              O: <span className="text-player-o font-medium ml-1">{gameState.players.O}</span>
            </p>
            {gameState.gameActive && (
              <p>
                Current Turn: <span className={`font-medium ${gameState.currentPlayer === 'X' ? 'text-player-x' : 'text-player-o'}`}>
                  Player {gameState.currentPlayer}
                </span>
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TicTacToeGame;
