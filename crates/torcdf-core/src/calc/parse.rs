//! Formula text to an expression tree.

use super::CalcError;

/// A parsed formula. `Var` holds the name as written (a backtick name
/// without its backticks); the checker resolves it to a path.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Func {
    Sqrt,
    Abs,
    Exp,
    Log,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Deg,
    Rad,
    Min,
    Max,
    Hypot,
    Floor,
    Ceil,
    Round,
    Where,
}

const ALL_FUNCS: [Func; 21] = [
    Func::Sqrt,
    Func::Abs,
    Func::Exp,
    Func::Log,
    Func::Log10,
    Func::Sin,
    Func::Cos,
    Func::Tan,
    Func::Asin,
    Func::Acos,
    Func::Atan,
    Func::Atan2,
    Func::Deg,
    Func::Rad,
    Func::Min,
    Func::Max,
    Func::Hypot,
    Func::Floor,
    Func::Ceil,
    Func::Round,
    Func::Where,
];

impl Func {
    pub fn name(self) -> &'static str {
        match self {
            Func::Sqrt => "sqrt",
            Func::Abs => "abs",
            Func::Exp => "exp",
            Func::Log => "log",
            Func::Log10 => "log10",
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Tan => "tan",
            Func::Asin => "asin",
            Func::Acos => "acos",
            Func::Atan => "atan",
            Func::Atan2 => "atan2",
            Func::Deg => "deg",
            Func::Rad => "rad",
            Func::Min => "min",
            Func::Max => "max",
            Func::Hypot => "hypot",
            Func::Floor => "floor",
            Func::Ceil => "ceil",
            Func::Round => "round",
            Func::Where => "where",
        }
    }

    /// How many values the function takes.
    pub fn arity(self) -> usize {
        match self {
            Func::Atan2 | Func::Min | Func::Max | Func::Hypot => 2,
            Func::Where => 3,
            _ => 1,
        }
    }

    pub fn from_name(s: &str) -> Option<Func> {
        ALL_FUNCS.into_iter().find(|f| f.name() == s)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Quoted(String),
    Op(&'static str),
    LParen,
    RParen,
    Comma,
    End,
}

impl Tok {
    fn text(&self) -> String {
        match self {
            Tok::Num(v) => v.to_string(),
            Tok::Ident(s) => s.clone(),
            Tok::Quoted(s) => format!("`{s}`"),
            Tok::Op(o) => (*o).to_string(),
            Tok::LParen => "(".into(),
            Tok::RParen => ")".into(),
            Tok::Comma => ",".into(),
            Tok::End => "end".into(),
        }
    }
}

/// Tokens with their byte positions; the last is `End`.
fn lex(src: &str) -> Result<Vec<(Tok, usize)>, CalcError> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        if c.is_ascii_digit() || (c == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                if !(i < b.len() && b[i].is_ascii_digit()) {
                    return Err(CalcError::at(start, format!("bad number {}", &src[start..i])));
                }
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let text = &src[start..i];
            let v: f64 = text.parse().map_err(|_| CalcError::at(start, format!("bad number {text}")))?;
            out.push((Tok::Num(v), start));
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push((Tok::Ident(src[start..i].to_string()), start));
            continue;
        }
        if c == b'`' {
            let Some(len) = src[i + 1..].find('`') else {
                return Err(CalcError::at(start, "a name in backticks is not closed"));
            };
            let name = &src[i + 1..i + 1 + len];
            if name.is_empty() {
                return Err(CalcError::at(start, "empty name in backticks"));
            }
            out.push((Tok::Quoted(name.to_string()), start));
            i += len + 2;
            continue;
        }
        let two = src.get(i..i + 2).unwrap_or("");
        let op2 = ["**", "<=", ">=", "==", "!="].into_iter().find(|o| *o == two);
        if let Some(o) = op2 {
            out.push((Tok::Op(o), start));
            i += 2;
            continue;
        }
        let tok = match c {
            b'+' => Tok::Op("+"),
            b'-' => Tok::Op("-"),
            b'*' => Tok::Op("*"),
            b'/' => Tok::Op("/"),
            b'^' => Tok::Op("^"),
            b'<' => Tok::Op("<"),
            b'>' => Tok::Op(">"),
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b',' => Tok::Comma,
            _ => {
                let ch = src[i..].chars().next().unwrap_or('?');
                return Err(CalcError::at(start, format!("unexpected {ch}")));
            }
        };
        out.push((tok, start));
        i += 1;
    }
    out.push((Tok::End, src.len()));
    Ok(out)
}

struct Parser {
    toks: Vec<(Tok, usize)>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.at].0
    }

    fn pos(&self) -> usize {
        self.toks[self.at].1
    }

    fn next(&mut self) -> Tok {
        let t = self.toks[self.at].0.clone();
        if self.at + 1 < self.toks.len() {
            self.at += 1;
        }
        t
    }

    fn is_word(&self, w: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == w)
    }

    fn unexpected(&self) -> CalcError {
        match self.peek() {
            Tok::End => CalcError::at(self.pos(), "the formula ends too early"),
            t => CalcError::at(self.pos(), format!("unexpected {}", t.text())),
        }
    }

    fn or(&mut self) -> Result<Expr, CalcError> {
        let mut e = self.and()?;
        while self.is_word("or") {
            self.next();
            e = Expr::Bin(BinOp::Or, Box::new(e), Box::new(self.and()?));
        }
        Ok(e)
    }

    fn and(&mut self) -> Result<Expr, CalcError> {
        let mut e = self.not()?;
        while self.is_word("and") {
            self.next();
            e = Expr::Bin(BinOp::And, Box::new(e), Box::new(self.not()?));
        }
        Ok(e)
    }

    fn not(&mut self) -> Result<Expr, CalcError> {
        if self.is_word("not") {
            self.next();
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        self.cmp()
    }

    fn cmp_op(&self) -> Option<BinOp> {
        match self.peek() {
            Tok::Op("<") => Some(BinOp::Lt),
            Tok::Op("<=") => Some(BinOp::Le),
            Tok::Op(">") => Some(BinOp::Gt),
            Tok::Op(">=") => Some(BinOp::Ge),
            Tok::Op("==") => Some(BinOp::Eq),
            Tok::Op("!=") => Some(BinOp::Ne),
            _ => None,
        }
    }

    fn cmp(&mut self) -> Result<Expr, CalcError> {
        let e = self.add()?;
        let Some(op) = self.cmp_op() else { return Ok(e) };
        self.next();
        let r = self.add()?;
        if self.cmp_op().is_some() {
            return Err(CalcError::at(self.pos(), "comparisons cannot be chained; use and"));
        }
        Ok(Expr::Bin(op, Box::new(e), Box::new(r)))
    }

    fn add(&mut self) -> Result<Expr, CalcError> {
        let mut e = self.mul()?;
        loop {
            let op = match self.peek() {
                Tok::Op("+") => BinOp::Add,
                Tok::Op("-") => BinOp::Sub,
                _ => return Ok(e),
            };
            self.next();
            e = Expr::Bin(op, Box::new(e), Box::new(self.mul()?));
        }
    }

    fn mul(&mut self) -> Result<Expr, CalcError> {
        let mut e = self.unary()?;
        loop {
            let op = match self.peek() {
                Tok::Op("*") => BinOp::Mul,
                Tok::Op("/") => BinOp::Div,
                _ => return Ok(e),
            };
            self.next();
            e = Expr::Bin(op, Box::new(e), Box::new(self.unary()?));
        }
    }

    fn unary(&mut self) -> Result<Expr, CalcError> {
        match self.peek() {
            Tok::Op("-") => {
                self.next();
                Ok(Expr::Neg(Box::new(self.unary()?)))
            }
            Tok::Op("+") => {
                self.next();
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr, CalcError> {
        let base = self.primary()?;
        if matches!(self.peek(), Tok::Op("^") | Tok::Op("**")) {
            self.next();
            // Right-associative; the exponent may carry its own sign.
            let exp = self.unary()?;
            return Ok(Expr::Bin(BinOp::Pow, Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<Expr, CalcError> {
        let pos = self.pos();
        match self.peek().clone() {
            Tok::Num(v) => {
                self.next();
                Ok(Expr::Num(v))
            }
            Tok::Quoted(s) => {
                self.next();
                Ok(Expr::Var(s))
            }
            Tok::Ident(s) if matches!(s.as_str(), "and" | "or" | "not") => Err(self.unexpected()),
            Tok::Ident(s) => {
                self.next();
                if *self.peek() != Tok::LParen {
                    return Ok(Expr::Var(s));
                }
                let f = Func::from_name(&s).ok_or_else(|| CalcError::at(pos, format!("unknown function {s}")))?;
                self.next();
                let mut args = Vec::new();
                if *self.peek() != Tok::RParen {
                    loop {
                        args.push(self.or()?);
                        if *self.peek() == Tok::Comma {
                            self.next();
                        } else {
                            break;
                        }
                    }
                }
                if *self.peek() != Tok::RParen {
                    return Err(self.unexpected());
                }
                self.next();
                if args.len() != f.arity() {
                    let n = f.arity();
                    let values = if n == 1 { "value" } else { "values" };
                    return Err(CalcError::at(pos, format!("{s} takes {n} {values}, not {}", args.len())));
                }
                Ok(Expr::Call(f, args))
            }
            Tok::LParen => {
                self.next();
                let e = self.or()?;
                if *self.peek() != Tok::RParen {
                    return Err(match self.peek() {
                        Tok::End => CalcError::at(pos, "a bracket is not closed"),
                        _ => self.unexpected(),
                    });
                }
                self.next();
                Ok(e)
            }
            _ => Err(self.unexpected()),
        }
    }
}

/// Parses formula text.
pub fn parse(src: &str) -> Result<Expr, CalcError> {
    if src.trim().is_empty() {
        return Err(CalcError::msg("the formula is empty"));
    }
    let mut p = Parser { toks: lex(src)?, at: 0 };
    let e = p.or()?;
    if *p.peek() != Tok::End {
        return Err(p.unexpected());
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(v: f64) -> Box<Expr> {
        Box::new(Expr::Num(v))
    }
    fn v(s: &str) -> Box<Expr> {
        Box::new(Expr::Var(s.into()))
    }
    fn bin(op: BinOp, a: Box<Expr>, b: Box<Expr>) -> Box<Expr> {
        Box::new(Expr::Bin(op, a, b))
    }

    #[test]
    fn precedence() {
        use BinOp::*;
        assert_eq!(parse("1+2*3^2").unwrap(), *bin(Add, n(1.0), bin(Mul, n(2.0), bin(Pow, n(3.0), n(2.0)))));
        assert_eq!(parse("-2^2").unwrap(), Expr::Neg(bin(Pow, n(2.0), n(2.0))));
        assert_eq!(parse("2^3^2").unwrap(), *bin(Pow, n(2.0), bin(Pow, n(3.0), n(2.0))));
        assert_eq!(parse("a**2").unwrap(), parse("a^2").unwrap());
        assert_eq!(parse("(1+2)*3").unwrap(), *bin(Mul, bin(Add, n(1.0), n(2.0)), n(3.0)));
        assert_eq!(parse("8/4/2").unwrap(), *bin(Div, bin(Div, n(8.0), n(4.0)), n(2.0)));
    }

    #[test]
    fn logic_and_compare() {
        use BinOp::*;
        let want = bin(
            Or,
            bin(And, bin(Gt, v("a"), n(1.0)), Box::new(Expr::Not(bin(Eq, v("b"), n(2.0))))),
            v("c"),
        );
        assert_eq!(parse("a > 1 and not b == 2 or c").unwrap(), *want);
        assert!(parse("a < b < c").is_err());
        assert_eq!(parse("a <= b").unwrap(), *bin(Le, v("a"), v("b")));
        assert_eq!(parse("a != b").unwrap(), *bin(Ne, v("a"), v("b")));
    }

    #[test]
    fn calls() {
        assert_eq!(parse("atan2(vo, uo)").unwrap(), Expr::Call(Func::Atan2, vec![Expr::Var("vo".into()), Expr::Var("uo".into())]));
        let e = parse("sqrt(1, 2)").unwrap_err();
        assert!(e.msg.contains("sqrt takes 1"), "{e}");
        assert_eq!(parse("foo(1)").unwrap_err().msg, "unknown function foo");
        assert_eq!(parse("where(a > 0, a, 0)").unwrap(), Expr::Call(Func::Where, vec![*bin(BinOp::Gt, v("a"), n(0.0)), Expr::Var("a".into()), Expr::Num(0.0)]));
    }

    #[test]
    fn names_that_look_like_functions() {
        let want = bin(BinOp::Add, v("max"), Box::new(Expr::Call(Func::Max, vec![Expr::Var("a".into()), Expr::Num(1.0)])));
        assert_eq!(parse("max + max(a, 1)").unwrap(), *want);
        assert_eq!(parse("pi").unwrap(), Expr::Var("pi".into()));
    }

    #[test]
    fn backticks() {
        assert_eq!(parse("`sea-level` * 2").unwrap(), *bin(BinOp::Mul, v("sea-level"), n(2.0)));
        assert_eq!(parse("`/ocean/uo`").unwrap(), Expr::Var("/ocean/uo".into()));
        assert_eq!(parse("`abc").unwrap_err().pos, Some(0));
    }

    #[test]
    fn numbers() {
        assert_eq!(parse("1e-4").unwrap(), Expr::Num(1e-4));
        assert_eq!(parse(".5").unwrap(), Expr::Num(0.5));
        assert_eq!(parse("2.").unwrap(), Expr::Num(2.0));
        assert!(parse("1e").is_err());
    }

    #[test]
    fn error_position() {
        let e = parse("a + * b").unwrap_err();
        assert_eq!(e.pos, Some(4));
        assert_eq!(e.to_string(), "unexpected * (at character 5)");
        assert!(parse("a +").is_err());
        assert!(parse("(a").is_err());
        assert!(parse("a b").is_err());
    }

    #[test]
    fn every_function_has_help() {
        for f in ALL_FUNCS {
            assert!(super::super::FUNCTIONS.iter().any(|(n, _)| *n == f.name()), "{f:?}");
            assert_eq!(Func::from_name(f.name()), Some(f));
        }
        assert_eq!(super::super::FUNCTIONS.len(), ALL_FUNCS.len());
    }

    #[test]
    fn empty() {
        assert_eq!(parse("  ").unwrap_err().msg, "the formula is empty");
    }
}
