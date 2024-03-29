# Public subnets

resource "aws_subnet" "public_subnet_a" {
  vpc_id                  = aws_vpc.main_vpc.id
  availability_zone       = "${var.aws_region}a"
  cidr_block              = cidrsubnet(aws_vpc.main_vpc.cidr_block, 4, 1)
  map_public_ip_on_launch = true

  ipv6_cidr_block                 = cidrsubnet(aws_vpc.main_vpc.ipv6_cidr_block, 8, 1)
  assign_ipv6_address_on_creation = true

  tags = {
    Name                                            = "${var.project}-public-subnet-${var.aws_region}a",
    Organization                                    = var.organization
    Project                                         = var.project
    "kubernetes.io/role/elb"                        = "1"
    "kubernetes.io/cluster/${var.eks_cluster_name}" = "shared"
  }
}

resource "aws_subnet" "public_subnet_b" {
  vpc_id                  = aws_vpc.main_vpc.id
  availability_zone       = "${var.aws_region}b"
  cidr_block              = cidrsubnet(aws_vpc.main_vpc.cidr_block, 4, 2)
  map_public_ip_on_launch = true

  ipv6_cidr_block                 = cidrsubnet(aws_vpc.main_vpc.ipv6_cidr_block, 8, 2)
  assign_ipv6_address_on_creation = true

  tags = {
    Name                                            = "${var.project}-public-subnet-${var.aws_region}b",
    Organization                                    = var.organization
    Project                                         = var.project
    "kubernetes.io/role/elb"                        = "1"
    "kubernetes.io/cluster/${var.eks_cluster_name}" = "shared"
  }
}

resource "aws_subnet" "public_subnet_c" {
  vpc_id                  = aws_vpc.main_vpc.id
  availability_zone       = "${var.aws_region}c"
  cidr_block              = cidrsubnet(aws_vpc.main_vpc.cidr_block, 4, 3)
  map_public_ip_on_launch = true

  ipv6_cidr_block                 = cidrsubnet(aws_vpc.main_vpc.ipv6_cidr_block, 8, 3)
  assign_ipv6_address_on_creation = true

  tags = {
    Name                                            = "${var.project}-public-subnet-${var.aws_region}c",
    Organization                                    = var.organization
    Project                                         = var.project
    "kubernetes.io/role/elb"                        = "1"
    "kubernetes.io/cluster/${var.eks_cluster_name}" = "shared"
  }
}


# Private subnets

resource "aws_subnet" "private_subnet_a" {
  vpc_id                  = aws_vpc.main_vpc.id
  availability_zone       = "${var.aws_region}a"
  cidr_block              = cidrsubnet(aws_vpc.main_vpc.cidr_block, 4, 4)
  map_public_ip_on_launch = false

  ipv6_cidr_block                 = cidrsubnet(aws_vpc.main_vpc.ipv6_cidr_block, 8, 4)
  assign_ipv6_address_on_creation = true

  tags = {
    Name                                            = "${var.project}-private-subnet-${var.aws_region}a",
    Organization                                    = var.organization
    Project                                         = var.project
    "kubernetes.io/role/internal-elb"               = "1"
    "kubernetes.io/cluster/${var.eks_cluster_name}" = "shared"
  }
}

resource "aws_subnet" "private_subnet_b" {
  vpc_id                  = aws_vpc.main_vpc.id
  availability_zone       = "${var.aws_region}b"
  cidr_block              = cidrsubnet(aws_vpc.main_vpc.cidr_block, 4, 5)
  map_public_ip_on_launch = false

  ipv6_cidr_block                 = cidrsubnet(aws_vpc.main_vpc.ipv6_cidr_block, 8, 5)
  assign_ipv6_address_on_creation = true

  tags = {
    Name                                            = "${var.project}-private-subnet-${var.aws_region}b",
    Organization                                    = var.organization
    Project                                         = var.project
    "kubernetes.io/role/internal-elb"               = "1"
    "kubernetes.io/cluster/${var.eks_cluster_name}" = "shared"
  }
}

resource "aws_subnet" "private_subnet_c" {
  vpc_id                  = aws_vpc.main_vpc.id
  availability_zone       = "${var.aws_region}c"
  cidr_block              = cidrsubnet(aws_vpc.main_vpc.cidr_block, 4, 6)
  map_public_ip_on_launch = false

  ipv6_cidr_block                 = cidrsubnet(aws_vpc.main_vpc.ipv6_cidr_block, 8, 6)
  assign_ipv6_address_on_creation = true

  tags = {
    Name                                            = "${var.project}-private-subnet-${var.aws_region}c",
    Organization                                    = var.organization
    Project                                         = var.project
    "kubernetes.io/role/internal-elb"               = "1"
    "kubernetes.io/cluster/${var.eks_cluster_name}" = "shared"
  }
}
